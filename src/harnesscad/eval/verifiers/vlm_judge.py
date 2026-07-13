"""VLM-as-judge — the subjective slice of verification.

docs/blueprint.md sec.6 is explicit: hard geometry (manifold, watertight,
dimensions, validity) is judged by the *kernel*, never the LLM. The LLM judge
is reserved for the **subjective slice** — design intent, cleanliness, "does the
shape read like the brief" — and stays **advisory**: it emits INFO/WARNING
diagnostics carrying a 0..1 score, never an ERROR that could block the loop.

This reuses the SpatialHero pattern (sec.21): render isometric + orthographic
views, hand them to a vision model with a rubric, parse a 0..1 reward. Two
bias-defence mechanisms from sec.6 are built in:

  * **Swap-augmentation.** Position bias is real, so when comparing candidate A
    vs B we judge both orderings (A,B) and (B,A) and average. For a single model
    we swap the *view order* across two passes and average, damping any
    ordering artefact. See :meth:`VLMJudgeCheck.compare` and ``swap_augment``.
  * **G-Eval.** Instead of one hard number, the judge can return a probability
    distribution over discrete score buckets; :class:`GEvalScore` collapses it
    to a probability-weighted expectation (finer-grained, less brittle).

Safety rails:
  * If nothing renders (no kernel / no solid / headless), the check INFO-*skips*
    with a clear message rather than failing.
  * The ground-truth answer is **never** put in the prompt — only the brief and
    rubric (both are legitimately part of the task, unlike a reference solution).
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from harnesscad.agents.llm.base import Message
from harnesscad.io.surfaces.render import DEFAULT_VIEWS, RenderResult, render
from harnesscad.eval.verifiers.verify import Diagnostic, Severity, VerifyReport


# --- G-Eval helper ---------------------------------------------------------
@dataclass
class GEvalScore:
    """Probability-weighted score (G-Eval, sec.6).

    Given a distribution over discrete score buckets (e.g. the judge's token
    probabilities for "1".."5"), the expected value is a smoother reward than a
    single argmax pick. ``scale`` normalises the bucket labels into 0..1.
    """

    distribution: Dict[float, float]
    scale: Tuple[float, float] = (0.0, 1.0)

    @property
    def value(self) -> float:
        lo, hi = self.scale
        span = (hi - lo) or 1.0
        total = sum(self.distribution.values())
        if total <= 0:
            return 0.0
        expected = sum(k * p for k, p in self.distribution.items()) / total
        return _clamp01((expected - lo) / span)

    @classmethod
    def from_distribution(cls, dist: Dict, scale: Tuple[float, float] = (0.0, 1.0)):
        clean: Dict[float, float] = {}
        for k, p in dist.items():
            try:
                clean[float(k)] = float(p)
            except (TypeError, ValueError):
                continue
        return cls(clean, scale)


# --- parsing helpers -------------------------------------------------------
@dataclass
class JudgeVerdict:
    score: float
    rationale: str = ""
    raw: Optional[dict] = None


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_json(text: str) -> JudgeVerdict:
    """Parse a judge reply into a 0..1 :class:`JudgeVerdict`.

    Tolerant: extracts the first JSON object from surrounding prose. Accepts a
    plain ``score`` (auto-normalised if it looks like a 0..10 / 0..100 scale) or
    a G-Eval ``distribution``/``scores`` map. Malformed input -> score 0.0.
    """
    obj: Optional[dict] = None
    if text:
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            m = _JSON_OBJ.search(text)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except (ValueError, TypeError):
                    obj = None
    if not isinstance(obj, dict):
        return JudgeVerdict(0.0, "unparseable judge response", None)

    rationale = str(obj.get("rationale") or obj.get("reason")
                    or obj.get("explanation") or "")

    dist = obj.get("distribution") or obj.get("scores")
    if isinstance(dist, dict):
        # G-Eval buckets are typically 1..5; infer scale from the labels.
        keys = []
        for k in dist:
            try:
                keys.append(float(k))
            except (TypeError, ValueError):
                pass
        scale = (min(keys), max(keys)) if len(keys) >= 2 else (0.0, 1.0)
        return JudgeVerdict(GEvalScore.from_distribution(dist, scale).value,
                            rationale, obj)

    raw_score = obj.get("score", obj.get("reward"))
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return JudgeVerdict(0.0, rationale or "no score field", obj)
    if score > 1.0:  # tolerate 0..10 / 0..100 rubrics
        score = score / (100.0 if score > 10.0 else 10.0)
    return JudgeVerdict(_clamp01(score), rationale, obj)


# --- prompt construction ---------------------------------------------------
DEFAULT_RUBRIC = (
    "Judge ONLY the subjective design quality of the rendered part against the "
    "brief: does the overall form read as the described object, is it clean and "
    "plausible, are proportions sensible. Do NOT judge exact dimensions, "
    "manifoldness or validity (the geometry kernel already checks those). "
    "Return STRICT JSON: {\"score\": <float 0..1>, \"rationale\": <short string>}. "
    "1.0 = clearly matches the brief's intent, 0.0 = unrelated or malformed."
)


def _views_to_data_uris(result: RenderResult) -> List[Tuple[str, str]]:
    """(view_name, data-uri) for each rendered view, in a stable order."""
    mime = "image/png" if result.fmt == "png" else "image/svg+xml"
    out: List[Tuple[str, str]] = []
    for name, data in result.images.items():
        if data is None:
            continue
        b64 = base64.b64encode(data).decode("ascii")
        out.append((name, f"data:{mime};base64,{b64}"))
    return out


def build_judge_messages(
    brief: str,
    rubric: str,
    view_order: List[str],
    data_uris: Dict[str, str],
) -> List[Message]:
    """System+user messages. Images are attached as data URIs in ``view_order``.

    Never includes a ground-truth/reference solution — only the brief + rubric.
    """
    system = Message("system",
                     "You are a meticulous CAD design reviewer acting as an "
                     "advisory judge. " + rubric)
    lines = [
        "DESIGN BRIEF:",
        brief.strip(),
        "",
        f"You are shown {len(view_order)} rendered views of a candidate model "
        "in this order: " + ", ".join(view_order) + ".",
    ]
    for name in view_order:
        uri = data_uris.get(name)
        if uri is not None:
            lines.append(f"[view:{name}] {uri}")
    lines.append("")
    lines.append("Score the model now as STRICT JSON.")
    return [system, Message("user", "\n".join(lines))]


# --- the verifier ----------------------------------------------------------
class VLMJudgeCheck:
    """Advisory vision-judge verifier (``name='vlm-judge'``).

    Implements the Verifier protocol (``check(backend, opdag) -> VerifyReport``).
    Renders the model, asks a vision-capable ``llm`` for a 0..1 score against the
    brief+rubric, and emits an INFO (score >= ``pass_threshold``) or WARNING
    (below) diagnostic carrying the score. It **never** emits ERROR — hard
    geometry stays with the kernel.
    """

    name = "vlm-judge"

    def __init__(
        self,
        llm,
        brief: str,
        rubric: Optional[str] = None,
        swap_augment: bool = True,
        views=DEFAULT_VIEWS,
        size: Tuple[int, int] = (512, 512),
        fmt: str = "svg",
        pass_threshold: float = 0.5,
        use_geval: bool = False,
    ) -> None:
        self.llm = llm
        self.brief = brief
        self.rubric = rubric or DEFAULT_RUBRIC
        self.swap_augment = swap_augment
        self.views = tuple(views)
        self.size = size
        self.fmt = fmt
        self.pass_threshold = pass_threshold
        self.use_geval = use_geval

    # -- one judged pass over a set of rendered views -----------------------
    def _judge_once(self, data_uris: Dict[str, str], view_order: List[str]) -> JudgeVerdict:
        messages = build_judge_messages(
            self.brief, self.rubric, view_order, data_uris)
        # Also pass raw ordered uris via opts so a real vision backend adapter
        # can attach them as image parts; text-only mocks simply ignore this.
        result = self.llm.complete(
            messages,
            temperature=0.0,
            images=[data_uris[n] for n in view_order if n in data_uris],
        )
        return parse_judge_json(result.text)

    def _judge(self, result: RenderResult) -> JudgeVerdict:
        """Judge one model, applying view-order swap-augmentation if enabled."""
        data_uris = dict(_views_to_data_uris(result))
        order = list(data_uris.keys())
        verdicts = [self._judge_once(data_uris, order)]
        if self.swap_augment and len(order) > 1:
            verdicts.append(self._judge_once(data_uris, list(reversed(order))))
        score = sum(v.score for v in verdicts) / len(verdicts)
        rationale = verdicts[0].rationale
        return JudgeVerdict(_clamp01(score), rationale,
                            {"passes": [v.raw for v in verdicts]})

    # -- Verifier protocol --------------------------------------------------
    def check(self, backend, opdag=None) -> VerifyReport:
        result = render(backend, views=self.views, size=self.size, fmt=self.fmt)
        if not result.any_rendered:
            return VerifyReport([Diagnostic(
                Severity.INFO, "vlm-judge-skip",
                f"vision judge skipped (no rendered views): {result.note}")])

        verdict = self._judge(result)
        sev = Severity.INFO if verdict.score >= self.pass_threshold else Severity.WARNING
        rationale = f" — {verdict.rationale}" if verdict.rationale else ""
        return VerifyReport([Diagnostic(
            sev, "vlm-judge",
            f"subjective design score {verdict.score:.3f} "
            f"(advisory; threshold {self.pass_threshold:.2f}){rationale}")])

    # -- A-vs-B comparison with full swap-augmentation ----------------------
    def compare(self, backend_a, backend_b) -> Dict[str, float]:
        """Judge candidate A vs B, defeating position bias by averaging both
        orderings (A,B) and (B,A). Returns per-candidate mean scores + which
        won. Falls back to independent scoring when a candidate cannot render.
        """
        ra = render(backend_a, views=self.views, size=self.size, fmt=self.fmt)
        rb = render(backend_b, views=self.views, size=self.size, fmt=self.fmt)
        if not ra.any_rendered or not rb.any_rendered:
            sa = self._judge(ra).score if ra.any_rendered else None
            sb = self._judge(rb).score if rb.any_rendered else None
            return {"a": sa, "b": sb, "winner": None, "skipped": True}

        # Two orderings; each is an independent judged pass -> average per side.
        va1, vb1 = self._judge(ra).score, self._judge(rb).score       # A then B
        vb2, va2 = self._judge(rb).score, self._judge(ra).score       # B then A
        sa = (va1 + va2) / 2.0
        sb = (vb1 + vb2) / 2.0
        winner = "a" if sa > sb else ("b" if sb > sa else "tie")
        return {"a": sa, "b": sb, "winner": winner, "skipped": False}
