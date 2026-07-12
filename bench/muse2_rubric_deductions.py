"""MUSE deterministic deduction-rule rubric engine (judge scaffolding).

Re-implements the *deterministic* half of the MUSE judge -- the part that runs
before any LLM/VLM is consulted -- from ``src/judge_system/rubric.py`` and
``data_prep.py`` of the muse-benchmark repository (Dong et al., "MUSE:
Benchmarking Manufacturable, Functional, and Assemblable Text-to-CAD
Generation").

Unlike ``bench/muse_scorecard`` (the three-stage funnel that aggregates already
graded sub-criteria) and the three pillar scorers, this module reproduces the
per-rubric-item *deduction* mechanism that turns raw geometry / sandbox / drawing
measurements into a 0-1 item score with no model in the loop:

  * every rubric item starts at a perfect score of 1.0;
  * each attached deduction rule has a ``rule_code`` (a named predicate over the
    measured metrics) and a ``deduction_ratio`` in [0, 1];
  * a rule that fires subtracts its ratio; the item score is clamped to [0, 1];
  * item scores are aggregated by normalized weight and by rubric category.

The rule predicates are the same ones the repo evaluates deterministically
(e.g. ``code_or_result_missing``, ``component_count_mismatch``,
``global_geometry_invalid``); the Chinese secondary-category strings are
replaced by ASCII rule codes. This is the deterministic scaffolding the campaign
is meant to keep; the natural-language rationale text a VLM would add is out of
scope.

Also provides the two deterministic spec parsers the repo uses to feed this
engine: ``expected_component_count_from_plan`` (reads the planned assembly count
from a plan document) and ``normalize_rubric_weights`` (dedup + weight
normalisation from ``data_prep._extract_rubric_items``).

No wall clock, no randomness.
"""

from __future__ import annotations

# --- Metrics context ---------------------------------------------------------
# A flat, injectable snapshot of everything the deterministic rules read. Every
# geometry flag is tri-state: True (checked, passed), False (checked, failed) or
# None (not evaluated -- e.g. code never ran). This mirrors the repo's
# GeometryMetrics tri-state semantics exactly.

_GEOMETRY_FLAGS = (
    "code_valid", "geometry_valid", "watertight", "manifold",
    "self_intersection_free", "normal_consistency", "volume_valid",
    "bbox_valid", "occt_valid",
)


class MetricsContext:
    """Injected deterministic measurements for one candidate design.

    Keyword fields (all optional; unspecified geometry flags default to None):
      code_valid, geometry_valid, watertight, manifold,
      self_intersection_free, normal_consistency, volume_valid, bbox_valid,
      occt_valid : tri-state bool | None geometry checks.
      sandbox_ok        : bool -- sandbox executed and exported a solid.
      bbox              : sequence of bounding-box extents (floats).
      solid_count       : int  -- number of solids the sandbox produced.
      svg_path_count    : int  -- <path> count in the 4-view drawing.
      svg_component_estimate : int -- components estimated from the drawing.
      expected_components    : int -- planned assembly count (ground truth).
    """

    __slots__ = (_GEOMETRY_FLAGS + (
        "sandbox_ok", "bbox", "solid_count", "svg_path_count",
        "svg_component_estimate", "expected_components"))

    def __init__(self, **kw):
        for name in _GEOMETRY_FLAGS:
            setattr(self, name, kw.get(name, None))
        self.sandbox_ok = bool(kw.get("sandbox_ok", False))
        self.bbox = tuple(kw.get("bbox", ()) or ())
        self.solid_count = int(kw.get("solid_count", 0) or 0)
        self.svg_path_count = int(kw.get("svg_path_count", 0) or 0)
        self.svg_component_estimate = int(kw.get("svg_component_estimate", 0) or 0)
        self.expected_components = int(kw.get("expected_components", 0) or 0)

    def bbox_missing_or_collapsed(self):
        """True iff fewer than two bounding-box extents are strictly positive.

        Matches ``rubric._bbox_missing_or_collapsed``: a solid needs at least two
        positive extents to be a real 3D body rather than a line/plane/point.
        """
        positive = [v for v in self.bbox if v and v > 0]
        return len(positive) < 2

    def actual_components(self):
        """Realised component count: sandbox solids, else drawing estimate."""
        return self.solid_count or self.svg_component_estimate


# --- Deterministic rule predicates ------------------------------------------
# Each predicate maps a MetricsContext to (triggered, evidence). The set mirrors
# rubric._should_apply_rule; codes are the repo's own rule_code strings.

def _rule_code_or_result_missing(ctx):
    a = ctx.actual_components()
    triggered = (ctx.code_valid is not True) or (not ctx.sandbox_ok) or a <= 0
    return triggered, "code_valid=%r sandbox_ok=%r actual=%d" % (
        ctx.code_valid, ctx.sandbox_ok, a)


def _rule_global_geometry_invalid(ctx):
    triggered = (ctx.geometry_valid is not True) or ctx.occt_valid is False
    return triggered, "geometry_valid=%r occt_valid=%r" % (
        ctx.geometry_valid, ctx.occt_valid)


def _rule_bbox_missing_or_collapsed(ctx):
    return ctx.bbox_missing_or_collapsed(), "bbox=%r" % (ctx.bbox,)


def _rule_functional_support_or_access_risk(ctx):
    a = ctx.actual_components()
    triggered = (ctx.geometry_valid is not True) or a <= 0 or ctx.svg_path_count <= 0
    return triggered, "geometry_valid=%r actual=%d paths=%d" % (
        ctx.geometry_valid, a, ctx.svg_path_count)


def _rule_contact_safety_risk(ctx):
    triggered = (ctx.geometry_valid is not True) or ctx.bbox_valid is False
    return triggered, "geometry_valid=%r bbox_valid=%r" % (
        ctx.geometry_valid, ctx.bbox_valid)


def _rule_structural_strength_risk(ctx):
    triggered = ((ctx.geometry_valid is not True) or ctx.volume_valid is False
                 or ctx.bbox_valid is False)
    return triggered, "geometry_valid=%r volume_valid=%r bbox_valid=%r" % (
        ctx.geometry_valid, ctx.volume_valid, ctx.bbox_valid)


def _rule_component_count_mismatch(ctx):
    a = ctx.actual_components()
    triggered = a != ctx.expected_components
    return triggered, "expected=%d actual=%d" % (ctx.expected_components, a)


def _rule_assembly_relationship_risk(ctx):
    delta = abs(ctx.actual_components() - ctx.expected_components)
    triggered = delta >= 1 or ctx.geometry_valid is False
    return triggered, "component_delta=%d geometry_valid=%r" % (delta, ctx.geometry_valid)


def _rule_functional_structure_broken(ctx):
    triggered = (ctx.geometry_valid is not True) or ctx.watertight is False
    return triggered, "geometry_valid=%r watertight=%r" % (
        ctx.geometry_valid, ctx.watertight)


def _rule_local_continuity_risk(ctx):
    triggered = (ctx.self_intersection_free is False
                 or ctx.normal_consistency is False or ctx.volume_valid is False)
    return triggered, "sif=%r normals=%r volume_valid=%r" % (
        ctx.self_intersection_free, ctx.normal_consistency, ctx.volume_valid)


def _rule_process_fit_risk(ctx):
    triggered = (ctx.self_intersection_free is False or ctx.normal_consistency is False
                 or ctx.volume_valid is False or ctx.bbox_valid is False)
    return triggered, "sif=%r normals=%r volume_valid=%r bbox_valid=%r" % (
        ctx.self_intersection_free, ctx.normal_consistency,
        ctx.volume_valid, ctx.bbox_valid)


def _rule_parameter_range_fragility(ctx):
    triggered = ((ctx.geometry_valid is not True) or ctx.volume_valid is False
                 or ctx.bbox_valid is False)
    return triggered, "geometry_valid=%r volume_valid=%r bbox_valid=%r" % (
        ctx.geometry_valid, ctx.volume_valid, ctx.bbox_valid)


def _rule_narrow_safe_range(ctx):
    triggered = ctx.bbox_missing_or_collapsed() or ctx.geometry_valid is False
    return triggered, "bbox=%r geometry_valid=%r" % (ctx.bbox, ctx.geometry_valid)


# Canonical rule registry.
RULES = {
    "code_or_result_missing": _rule_code_or_result_missing,
    "global_geometry_invalid": _rule_global_geometry_invalid,
    "bbox_missing_or_collapsed": _rule_bbox_missing_or_collapsed,
    "functional_support_or_access_risk": _rule_functional_support_or_access_risk,
    "contact_safety_risk": _rule_contact_safety_risk,
    "structural_strength_risk": _rule_structural_strength_risk,
    "component_count_mismatch": _rule_component_count_mismatch,
    "assembly_relationship_risk": _rule_assembly_relationship_risk,
    "functional_structure_broken": _rule_functional_structure_broken,
    "local_continuity_risk": _rule_local_continuity_risk,
    "process_fit_risk": _rule_process_fit_risk,
    "parameter_range_fragility": _rule_parameter_range_fragility,
    "narrow_safe_range": _rule_narrow_safe_range,
}


def evaluate_rule(rule_code, ctx):
    """Evaluate one rule against a context. Unknown codes never trigger."""
    fn = RULES.get(str(rule_code).strip())
    if fn is None:
        return False, "rule_not_registered"
    return fn(ctx)


def score_item(item, ctx):
    """Deterministic 0-1 score for one rubric item.

    item : dict with keys
        item_id, title (opt), max_points (opt, default 1.0),
        normalized_weight (opt, default 0.0),
        deduction_rules : iterable of {rule_code, deduction_ratio, trigger?}.
    ctx  : MetricsContext.

    Returns dict: item_id, score (0-1 ratio), points (max_points*score),
    weight, deductions (list), rationale.
    """
    score = 1.0
    deductions = []
    for rule in item.get("deduction_rules", ()):
        rule_code = str(rule.get("rule_code", "")).strip()
        if not rule_code:
            continue
        triggered, evidence = evaluate_rule(rule_code, ctx)
        if not triggered:
            continue
        ratio = float(rule.get("deduction_ratio", 0.0) or 0.0)
        score -= ratio
        deductions.append({
            "rule_code": rule_code,
            "deduction_ratio": ratio,
            "trigger": rule.get("trigger", ""),
            "evidence": evidence,
        })
    score = max(0.0, min(1.0, score))
    if deductions:
        rationale = " | ".join(
            "%s: -%.2f (%s)" % (d["rule_code"], d["deduction_ratio"], d["evidence"])
            for d in deductions)
    else:
        rationale = "No deductions triggered."
    max_points = float(item.get("max_points", 1.0))
    return {
        "item_id": item.get("item_id", ""),
        "title": item.get("title", ""),
        "primary_category": item.get("primary_category", ""),
        "score": score,
        "points": max_points * score,
        "max_points": max_points,
        "weight": float(item.get("normalized_weight", 0.0)),
        "deductions": deductions,
        "rationale": rationale,
    }


def score_rubric(items, ctx):
    """Score every rubric item against one context (list of score_item dicts)."""
    return [score_item(item, ctx) for item in items]


def weighted_rubric_score(scored):
    """Weight-combined rubric score = sum(weight * score) over items."""
    return sum(s["weight"] * s["score"] for s in scored)


def category_breakdown(scored):
    """Aggregate scored items by ``primary_category``.

    Returns {category: {item_count, max_points, earned_points, weighted_score,
    ratio}} where ratio = earned_points / max_points (0 if no points).
    """
    out = {}
    for s in scored:
        cat = s.get("primary_category", "") or "(uncategorised)"
        b = out.setdefault(cat, {"item_count": 0, "max_points": 0.0,
                                 "earned_points": 0.0, "weighted_score": 0.0})
        b["item_count"] += 1
        b["max_points"] += s["max_points"]
        b["earned_points"] += s["points"]
        b["weighted_score"] += s["weight"] * s["score"]
    for b in out.values():
        b["ratio"] = 0.0 if b["max_points"] <= 0 else b["earned_points"] / b["max_points"]
    return out


# --- Deterministic spec parsers ----------------------------------------------

def expected_component_count_from_plan(plan_text, count_marker="## planned assembly count"):
    """Planned assembly-component count from a plan document.

    Mirrors ``rubric._expected_component_count_from_plan``: if a section marked
    by ``count_marker`` is present, the first standalone integer line after it is
    the count; otherwise fall back to the number of ``### `` sub-headings (one
    heading per planned component). ``count_marker`` matching is case-folded and
    whitespace-insensitive so both English and localised markers work.
    """
    lines = plan_text.splitlines()
    marker = "".join(count_marker.lower().split())
    in_section = False
    for line in lines:
        stripped = line.strip()
        if "".join(stripped.lower().split()).startswith(marker):
            in_section = True
            continue
        if in_section and stripped.lstrip("-").strip().isdigit():
            return int(stripped.lstrip("-").strip())
    headings = [ln for ln in lines if ln.startswith("### ")]
    return len(headings)


def normalize_rubric_weights(items):
    """Dedup rubric items and normalise raw weights to sum to 1.

    Mirrors ``data_prep._extract_rubric_items`` tail logic:
      * dedup by (title, description) keeping the largest ``raw_weight``;
      * if any positive-weight items remain, drop the zero-weight ones;
      * normalise so ``normalized_weight`` sums to 1 (uniform if all weights 0).

    items : iterable of dicts with at least title, description, raw_weight.
    Returns a new list of dicts with ``normalized_weight`` populated, in
    deterministic first-seen order.
    """
    deduped = {}
    order = []
    for it in items:
        key = (it.get("title", ""), it.get("description", ""))
        raw = float(it.get("raw_weight", 0.0) or 0.0)
        cur = deduped.get(key)
        if cur is None:
            order.append(key)
            deduped[key] = dict(it, raw_weight=raw)
        elif raw > float(cur.get("raw_weight", 0.0) or 0.0):
            deduped[key] = dict(it, raw_weight=raw)
    kept = [deduped[k] for k in order]
    positive = [it for it in kept if float(it.get("raw_weight", 0.0) or 0.0) > 0]
    if positive:
        kept = positive
    total = sum(float(it.get("raw_weight", 0.0) or 0.0) for it in kept)
    if total <= 0:
        total = float(len(kept)) or 1.0
        return [dict(it, normalized_weight=1.0 / total) for it in kept]
    return [dict(it, normalized_weight=float(it.get("raw_weight", 0.0) or 0.0) / total)
            for it in kept]
