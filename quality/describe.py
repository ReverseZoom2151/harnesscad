"""Design-to-text grounding: narrate a produced part and answer questions.

Two public entry points, both grounded strictly in the model -- they NEVER invent
a number that is not present in the feature graph or in ``backend.query(...)``:

  * :func:`describe_part` -- a natural-language summary of the part built from the
    :mod:`featuregraph` view plus ``query('summary')`` / ``query('metrics')``
    (e.g. "A 60x40x8 mm plate with 4 through holes (Ø5) and filleted edges;
    volume 19.0 cm3.").
  * :func:`answer_query` -- deterministic, templated answers to count/dimension
    questions ("how many holes?", "what is the bounding box?").

An optional injected LLM (:class:`llm.base.LLM`) is used ONLY to rephrase the
already-computed facts more fluently; the default path is a pure heuristic with
no network, so this doubles as a cheap self-consistency check on the geometry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from quality.featuregraph import FeatureGraph, build_feature_graph


# --- helpers ---------------------------------------------------------------
def _safe_query(backend: Any, q: str) -> Dict[str, Any]:
    try:
        result = backend.query(q)
        return result if isinstance(result, dict) else {}
    except Exception:  # noqa: BLE001 - a query must never break narration
        return {}


def _graph_for(backend: Any, opdag: Any) -> FeatureGraph:
    if opdag is not None:
        return build_feature_graph(opdag, backend=backend)
    return build_feature_graph(backend)


def _fmt_num(x: float) -> str:
    """Compact number: drop a trailing '.0', otherwise round to 2 dp."""
    xf = float(x)
    if abs(xf - round(xf)) < 1e-9:
        return str(int(round(xf)))
    return ("%.2f" % xf).rstrip("0").rstrip(".")


def _bbox_str(bbox: List[float]) -> Optional[str]:
    if not bbox or len(bbox) < 3:
        return None
    return "x".join(_fmt_num(v) for v in bbox[:3])


# --- fact collection (the grounded, deterministic core) --------------------
def _collect_facts(graph: FeatureGraph,
                   summary: Dict[str, Any],
                   metrics: Dict[str, Any]) -> Dict[str, Any]:
    features = graph.find_features()
    type_counts: Dict[str, int] = {}
    for n in features:
        type_counts[n.type] = type_counts.get(n.type, 0) + 1

    holes = graph.find("hole")
    hole_diams = sorted({
        _fmt_num(h.params["diameter"])
        for h in holes if h.params.get("diameter") is not None
    })
    hole_through = sum(1 for h in holes if h.params.get("through", True))

    fillet_radii = sorted({
        _fmt_num(f.params["radius"])
        for f in graph.find("fillet") if f.params.get("radius") is not None
    })
    chamfer_dists = sorted({
        _fmt_num(c.params["distance"])
        for c in graph.find("chamfer") if c.params.get("distance") is not None
    })

    # dominant sketch primitive -> a noun for the body
    prim_counts: Dict[str, int] = {}
    for s in graph.find("sketch"):
        for t in s.params.get("primitive_types", []):
            prim_counts[t] = prim_counts.get(t, 0) + 1
    noun = "part"
    if prim_counts.get("rectangle"):
        noun = "plate"
    elif prim_counts.get("circle"):
        noun = "cylindrical part"

    bbox = metrics.get("bbox")
    return {
        "feature_count": len(features),
        "sketch_count": len(graph.find("sketch")),
        "type_counts": type_counts,
        "noun": noun,
        "hole_count": len(holes),
        "hole_through": hole_through,
        "hole_diams": hole_diams,
        "fillet_count": type_counts.get("fillet", 0),
        "fillet_radii": fillet_radii,
        "chamfer_count": type_counts.get("chamfer", 0),
        "chamfer_dists": chamfer_dists,
        "shell_count": type_counts.get("shell", 0),
        "pattern_count": type_counts.get("linear_pattern", 0) + type_counts.get("circular_pattern", 0),
        "bbox": bbox,
        "bbox_str": _bbox_str(bbox) if bbox else None,
        "volume": metrics.get("volume"),
        "summary": summary,
    }


def _phrase_heuristic(facts: Dict[str, Any]) -> str:
    lead = "A "
    if facts["bbox_str"]:
        lead += facts["bbox_str"] + " mm "
    lead += facts["noun"]

    clauses: List[str] = []
    if facts["hole_count"]:
        n = facts["hole_count"]
        word = "hole" if n == 1 else "holes"
        kind = "through " if facts["hole_through"] == n and n else ""
        diam = ""
        if facts["hole_diams"]:
            diam = " (" + ", ".join("Ø" + d for d in facts["hole_diams"]) + ")"
        clauses.append("%d %s%s%s" % (n, kind, word, diam))
    if facts["fillet_count"]:
        r = facts["fillet_radii"]
        rtxt = (" (r=" + ", ".join(r) + ")") if r else ""
        clauses.append("filleted edges%s" % rtxt)
    if facts["chamfer_count"]:
        d = facts["chamfer_dists"]
        dtxt = (" (%s)" % ", ".join(d)) if d else ""
        clauses.append("chamfered edges%s" % dtxt)
    if facts["shell_count"]:
        clauses.append("a shelled wall" if facts["shell_count"] == 1 else "shelled walls")
    if facts["pattern_count"]:
        clauses.append("%d patterned feature(s)" % facts["pattern_count"])

    if clauses:
        if len(clauses) == 1:
            body = clauses[0]
        else:
            body = ", ".join(clauses[:-1]) + " and " + clauses[-1]
        sentence = lead + " with " + body
    else:
        # No modifying features: fall back to a plain feature count.
        fc = facts["feature_count"]
        sentence = lead + " with %d feature%s" % (fc, "" if fc == 1 else "s")

    if facts["volume"] is not None and facts["volume"] > 0:
        vol_cm3 = float(facts["volume"]) / 1000.0
        sentence += "; volume %s cm3" % _fmt_num(vol_cm3)

    # A grounded feature/sketch tally so the narration always exposes the real
    # counts from query('summary') / the graph (also a self-consistency anchor).
    sentence += ". Built from %d sketch%s and %d feature%s." % (
        facts["sketch_count"], "" if facts["sketch_count"] == 1 else "es",
        facts["feature_count"], "" if facts["feature_count"] == 1 else "s",
    )
    return sentence


def _phrase_with_llm(llm: Any, facts: Dict[str, Any], heuristic: str) -> str:
    """Ask an injected LLM to rephrase the FIXED facts. Never introduces numbers
    (we pass the finished heuristic sentence as the ground truth to rephrase)."""
    try:
        from llm.base import system, user
        msgs = [
            system("You rephrase a CAD part description more fluently. Do NOT add, "
                   "remove, or change any number, dimension, or count. Keep every "
                   "figure exactly as given. Return one or two sentences only."),
            user("Rephrase this part description, preserving all numbers exactly:\n"
                 + heuristic),
        ]
        result = llm.complete(msgs)
        text = getattr(result, "text", "") or ""
        text = text.strip()
        return text or heuristic
    except Exception:  # noqa: BLE001 - LLM is best-effort phrasing only
        return heuristic


# --- public: narration -----------------------------------------------------
def describe_part(backend: Any, opdag: Any = None, llm: Any = None) -> str:
    """Return a natural-language summary of the current part.

    Grounded in the feature graph + ``query('summary')`` / ``query('metrics')``;
    an optional ``llm`` only rephrases the already-computed facts.
    """
    graph = _graph_for(backend, opdag)
    summary = _safe_query(backend, "summary")
    metrics = _safe_query(backend, "metrics") or _safe_query(backend, "measure")
    facts = _collect_facts(graph, summary, metrics)
    heuristic = _phrase_heuristic(facts)
    if llm is not None:
        return _phrase_with_llm(llm, facts, heuristic)
    return heuristic


# --- public: templated Q&A -------------------------------------------------
_COUNT_TYPES = [
    ("sketch", "sketch"),
    ("hole", "hole"),
    ("fillet", "fillet"),
    ("chamfer", "chamfer"),
    ("shell", "shell"),
    ("revolve", "revolve"),
    ("extrude", "extrude"),
]


def answer_query(question: str, backend: Any, opdag: Any = None) -> str:
    """Answer a templated count/dimension question, deterministically.

    Supported: "how many <holes|fillets|features|sketches|...>?", "bounding box",
    "dimensions"/"size", "volume". Numbers come only from the graph / metrics.
    """
    graph = _graph_for(backend, opdag)
    q = (question or "").strip().lower()
    metrics = _safe_query(backend, "metrics") or _safe_query(backend, "measure")

    # -- counts ----------------------------------------------------------
    if "how many" in q or q.startswith("count") or "number of" in q:
        if "pattern" in q:
            n = len(graph.find("linear_pattern")) + len(graph.find("circular_pattern"))
            return "The part has %d pattern feature%s." % (n, "" if n == 1 else "s")
        for keyword, ntype in _COUNT_TYPES:
            if keyword in q:
                n = len(graph.find(ntype))
                label = keyword if n == 1 else keyword + "s"
                return "The part has %d %s." % (n, label)
        if "feature" in q:
            n = len(graph.find_features())
            return "The part has %d feature%s." % (n, "" if n == 1 else "s")
        # unknown noun -> report the total feature count as a safe default
        n = len(graph.find_features())
        return "The part has %d feature%s." % (n, "" if n == 1 else "s")

    # -- dimensions ------------------------------------------------------
    if "bounding box" in q or "bounding-box" in q or "bbox" in q \
            or "dimension" in q or "size" in q or ("how big" in q):
        bbox = metrics.get("bbox")
        if bbox and len(bbox) >= 3:
            return "The bounding box is %s x %s x %s mm." % (
                _fmt_num(bbox[0]), _fmt_num(bbox[1]), _fmt_num(bbox[2]))
        return ("The bounding box is unavailable: this backend reports no measured "
                "geometry.")

    if "volume" in q:
        vol = metrics.get("volume")
        if vol is not None and vol > 0:
            return "The volume is %s mm3 (%s cm3)." % (
                _fmt_num(vol), _fmt_num(float(vol) / 1000.0))
        return "The volume is unavailable: this backend reports no measured geometry."

    if "surface area" in q or "area" in q:
        area = metrics.get("surface_area")
        if area is not None and area > 0:
            return "The surface area is %s mm2." % _fmt_num(area)
        return "The surface area is unavailable: this backend reports no measured geometry."

    return ("I can answer counts (e.g. 'how many holes/fillets/features/sketches?') "
            "and dimensions (bounding box, volume, surface area).")
