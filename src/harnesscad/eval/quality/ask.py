"""ask — a natural-language question-answering layer over a built part.

The QA sibling of :func:`quality.describe.answer_query`: map a free-text
question to a *structured query* over the real model — the backend's read-only
``query('summary')`` / ``query('metrics')``, a :mod:`quality.featuregraph`
traversal, and a :mod:`quality.estimate` mass/cost estimate — and return a
deterministic, templated answer built **only** from figures that are actually in
the model. It never invents a number.

Recognised question shapes:

  * ``"how many holes?"`` / ``"number of fillets"``       -> feature counts
  * ``"total mass?"`` / ``"how heavy is it?"``            -> estimated mass (g)
  * ``"total volume?"`` / ``"bounding box?"``             -> measured metrics
  * ``"which holes < 5mm?"`` / ``"holes under 5 mm"``     -> filtered hole list
  * ``"list hole diameters"``                              -> the hole diameters

Anything unrecognised degrades gracefully: with an injected
:class:`llm.base.LLM` the question is rephrased/answered from the *already
computed* model facts (the LLM never sees a made-up number); without one, the
deterministic guidance string from :func:`quality.describe.answer_query` is
returned. No network on the default path.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from harnesscad.eval.quality.describe import answer_query, _fmt_num, _graph_for, _safe_query
from harnesscad.eval.quality.estimate import estimate_part

_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")


def ask(question: str,
        backend: Any,
        opdag: Any = None,
        llm: Any = None) -> str:
    """Answer ``question`` about the built part, grounded strictly in the model.

    ``backend`` supplies read-only queries + mass properties; ``opdag`` (optional)
    builds the richer feature graph; ``llm`` (optional) is used only to phrase an
    otherwise-unrecognised question from the computed facts — never to invent one.
    """
    q = (question or "").strip().lower()
    if not q:
        return ("Ask about counts (holes, fillets, features), dimensions "
                "(bounding box, volume), mass, or hole diameters.")

    graph = _graph_for(backend, opdag)

    # -- hole filtering: "which holes < 5mm?", "holes under 5 mm" ----------- #
    if "hole" in q and _is_threshold_query(q):
        return _answer_holes_threshold(q, graph)

    # -- hole diameter listing: "list hole diameters" ---------------------- #
    if "hole" in q and ("diameter" in q or "diameters" in q) \
            and ("list" in q or "what" in q or "which" in q):
        return _answer_hole_diameters(graph)

    # -- mass / weight: "total mass?", "how heavy?" ------------------------ #
    if "mass" in q or "weight" in q or "how heavy" in q:
        return _answer_mass(backend)

    # -- everything else: delegate to the deterministic templated Q&A ------ #
    ans = answer_query(question, backend, opdag)
    if _is_guidance(ans) and llm is not None:
        phrased = _answer_with_llm(llm, question, graph, backend, opdag)
        if phrased:
            return phrased
    return ans


# --------------------------------------------------------------------------- #
# Hole queries
# --------------------------------------------------------------------------- #
def _is_threshold_query(q: str) -> bool:
    if not _NUM_RE.search(q):
        return False
    return any(k in q for k in
               ("<", "less than", "under", "below", "smaller",
                ">", "greater than", "larger", "bigger", "over", "above"))


def _answer_holes_threshold(q: str, graph: Any) -> str:
    m = _NUM_RE.search(q)
    if not m:
        return _answer_hole_diameters(graph)
    thresh = float(m.group(1))
    greater = any(k in q for k in
                  (">", "greater than", "larger", "bigger", "over", "above"))
    holes = graph.find("hole")
    picked = []
    for h in holes:
        d = h.params.get("diameter")
        if d is None:
            continue
        if (d > thresh) if greater else (d < thresh):
            picked.append((h.id, float(d)))
    rel = "greater than" if greater else "less than"
    if not picked:
        return ("No holes with diameter %s %s mm (%d hole%s total)." % (
            rel, _fmt_num(thresh), len(holes), "" if len(holes) == 1 else "s"))
    listing = ", ".join("%s (Ø%s)" % (hid, _fmt_num(d)) for hid, d in picked)
    return "%d hole%s with diameter %s %s mm: %s." % (
        len(picked), "" if len(picked) == 1 else "s", rel,
        _fmt_num(thresh), listing)


def _answer_hole_diameters(graph: Any) -> str:
    holes = graph.find("hole")
    diams = [float(h.params["diameter"]) for h in holes
             if h.params.get("diameter") is not None]
    if not diams:
        if holes:
            return "The %d hole(s) report no diameter." % len(holes)
        return "The part has no holes."
    listing = ", ".join(_fmt_num(d) + " mm" for d in sorted(diams))
    return "Hole diameters (%d): %s." % (len(diams), listing)


# --------------------------------------------------------------------------- #
# Mass
# --------------------------------------------------------------------------- #
def _answer_mass(backend: Any) -> str:
    try:
        est = estimate_part(backend)
    except Exception:  # noqa: BLE001 - estimate must never break the answer
        est = None
    if est is None or not est.measured or est.mass is None:
        return ("The mass is unavailable: this backend reports no measured "
                "geometry ('metrics'/'measure').")
    grams = float(est.mass)
    return "The estimated mass is %s g (%s kg) in %s." % (
        _fmt_num(grams), _fmt_num(grams / 1000.0), est.material)


# --------------------------------------------------------------------------- #
# Fallback phrasing
# --------------------------------------------------------------------------- #
def _is_guidance(ans: str) -> bool:
    return ans.startswith("I can answer")


def _collect_fact_lines(graph: Any, backend: Any, opdag: Any) -> List[str]:
    facts: List[str] = []
    features = graph.find_features()
    facts.append("features: %d" % len(features))
    type_counts: dict = {}
    for n in features:
        type_counts[n.type] = type_counts.get(n.type, 0) + 1
    for t, c in sorted(type_counts.items()):
        facts.append("%s: %d" % (t, c))
    metrics = _safe_query(backend, "metrics") or _safe_query(backend, "measure")
    bbox = metrics.get("bbox")
    if bbox and len(bbox) >= 3:
        facts.append("bounding box mm: %s x %s x %s" % (
            _fmt_num(bbox[0]), _fmt_num(bbox[1]), _fmt_num(bbox[2])))
    if metrics.get("volume"):
        facts.append("volume mm3: %s" % _fmt_num(metrics["volume"]))
    return facts


def _answer_with_llm(llm: Any, question: str, graph: Any,
                     backend: Any, opdag: Any) -> Optional[str]:
    """Let an injected LLM phrase an answer from the FIXED, model-derived facts.

    The LLM is handed the grounded facts and told to answer only from them, so it
    can phrase an unrecognised question without ever inventing a figure.
    """
    try:
        from harnesscad.agents.llm.base import system, user
        facts = _collect_fact_lines(graph, backend, opdag)
        msgs = [
            system("You answer a question about a CAD part using ONLY the facts "
                   "given. Do not invent or estimate any number not listed. If "
                   "the facts do not contain the answer, say so plainly."),
            user("Facts:\n" + "\n".join(facts) + "\n\nQuestion: " + question),
        ]
        result = llm.complete(msgs)
        text = (getattr(result, "text", "") or "").strip()
        return text or None
    except Exception:  # noqa: BLE001 - LLM is best-effort only
        return None
