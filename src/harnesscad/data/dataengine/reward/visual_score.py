"""CADFusion multi-aspect visual scoring rubric (Section 3.3, Figure 4).

Implements the deterministic core of the LVM "harsh grader" that CADFusion
(Wang et al., ICML 2025, "Text-to-CAD Generation Through Infusing Visual
Feedback in Large Language Models") uses to turn a rendered CAD object into a
visual-feedback score. The LVM prompt (Appendix C.2, Listing 4) asks the grader
to comment on three aspects and then give a score out of 10:

  1. shape quality    -- regularity, naturalness and realism of the design;
  2. shape quantity   -- whether the number of components matches the
                         description (especially circular holes);
  3. distribution     -- whether components are arranged naturally, i.e. NOT
                         clustered together / colliding, and NOT excessively
                         spaced apart.

The LVM itself is external. What is deterministic and locally buildable is the
rubric that maps a rendered object's *geometry* (a set of axis-aligned component
boxes) plus the expected component count into the three sub-scores and the final
0-10 grade.

This is a genuinely distinct visual-feedback SIGNAL from the ones already in the
repo: it is not gift's single IoU-against-ground-truth (dataengine.gift_*), not
cadrille's execution reward, and not CADCodeVerify / Query2CAD's textual Q&A or
caption feedback. Here the signal is a three-criterion geometric grade over the
rendered object alone (no ground-truth object is required for quality and
distribution -- only the expected count is compared, mirroring the LVM which
sees the image and the text description, not the ground-truth mesh).

Deterministic, stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass

# Default rubric configuration.
DEFAULT_WEIGHTS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)  # (quality, quantity, distribution)
MAX_SCORE = 10.0
EPS = 1e-9

# Quality thresholds.
ASPECT_LIMIT = 20.0   # extent ratio above which a component reads as an unnatural sliver.

# Distribution thresholds.
SPACING_LIMIT = 3.0   # nearest-neighbour gap (in units of component size) beyond
                      # which spacing reads as "excessive".


@dataclass(frozen=True)
class Component:
    """One rendered CAD component as an axis-aligned bounding box in 3D.

    ``lo`` and ``hi`` are (x, y, z) tuples with lo[k] <= hi[k]. A component with
    zero extent in any axis is *degenerate* (a sliver/flat artefact the grader
    penalises under shape quality).
    """

    lo: tuple
    hi: tuple

    def __post_init__(self):
        if len(self.lo) != 3 or len(self.hi) != 3:
            raise ValueError("lo and hi must be 3-tuples")
        for a, b in zip(self.lo, self.hi):
            if b < a:
                raise ValueError("require lo[k] <= hi[k] on every axis")

    @property
    def extents(self):
        return tuple(float(b - a) for a, b in zip(self.lo, self.hi))

    @property
    def center(self):
        return tuple(0.5 * (a + b) for a, b in zip(self.lo, self.hi))

    @property
    def volume(self):
        ex, ey, ez = self.extents
        return ex * ey * ez

    @property
    def is_degenerate(self):
        return any(e <= EPS for e in self.extents)


def _clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def shape_quality_score(components, aspect_limit=ASPECT_LIMIT):
    """Criterion 1: regularity / naturalness / realism, in [0, 1].

    A component is well-formed when it is non-degenerate (non-zero volume) and
    not an extreme sliver (max/min extent ratio within ``aspect_limit``). The
    score is the fraction of well-formed components. An empty object scores 0.0
    (nothing was drawn).
    """
    comps = list(components)
    if not comps:
        return 0.0
    good = 0
    for c in comps:
        if c.is_degenerate:
            continue
        ex = c.extents
        ratio = max(ex) / min(ex)
        if ratio <= aspect_limit:
            good += 1
    return good / len(comps)


def shape_quantity_score(actual_count, expected_count):
    """Criterion 2: does the component count match the description, in [0, 1].

    1.0 on an exact match; decays linearly with the absolute count error,
    normalised by the expected count. If the description expects zero components
    the score is 1.0 only when none were drawn.
    """
    actual = int(actual_count)
    expected = int(expected_count)
    if actual < 0 or expected < 0:
        raise ValueError("counts must be non-negative")
    if expected == 0:
        return 1.0 if actual == 0 else 0.0
    return _clamp01(1.0 - abs(actual - expected) / expected)


def _overlap_volume(a, b):
    """Axis-aligned intersection volume of two components (0 if disjoint)."""
    vol = 1.0
    for k in range(3):
        lo = max(a.lo[k], b.lo[k])
        hi = min(a.hi[k], b.hi[k])
        d = hi - lo
        if d <= 0.0:
            return 0.0
        vol *= d
    return vol


def collision_penalty(components):
    """Mean pairwise overlap ratio in [0, 1]: components clustered / colliding.

    For each pair, the intersection volume is normalised by the smaller
    component's volume, then averaged over all pairs. 0.0 when nothing overlaps.
    """
    comps = [c for c in components if not c.is_degenerate]
    n = len(comps)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            inter = _overlap_volume(comps[i], comps[j])
            denom = min(comps[i].volume, comps[j].volume)
            total += (inter / denom) if denom > EPS else 0.0
            pairs += 1
    return _clamp01(total / pairs)


def _center_distance(a, b):
    ca, cb = a.center, b.center
    return sum((ca[k] - cb[k]) ** 2 for k in range(3)) ** 0.5


def _typical_size(c):
    """Characteristic length of a component (mean of its extents)."""
    return sum(c.extents) / 3.0


def spacing_penalty(components, spacing_limit=SPACING_LIMIT):
    """Mean excessive-spacing penalty in [0, 1]: components too far apart.

    For each component, the nearest-neighbour centre distance is measured in
    units of the pair's mean size. Gaps beyond ``spacing_limit`` are penalised,
    scaled so the penalty saturates at twice the limit. 0.0 for a single (or no)
    component -- spacing is only meaningful between objects.
    """
    comps = [c for c in components if not c.is_degenerate]
    n = len(comps)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        nearest = None
        for j in range(n):
            if i == j:
                continue
            size = 0.5 * (_typical_size(comps[i]) + _typical_size(comps[j]))
            gap = _center_distance(comps[i], comps[j]) / size if size > EPS else 0.0
            if nearest is None or gap < nearest:
                nearest = gap
        excess = (nearest - spacing_limit) / spacing_limit if nearest is not None else 0.0
        total += _clamp01(excess)
    return _clamp01(total / n)


def distribution_score(components, spacing_limit=SPACING_LIMIT):
    """Criterion 3: natural arrangement, in [0, 1].

    Penalised by both clustering/collisions and excessive spacing:
    ``1 - collision_penalty - spacing_penalty`` clamped to [0, 1]. A single
    component (nothing to arrange) scores 1.0.
    """
    comps = [c for c in components if not c.is_degenerate]
    if len(comps) < 2:
        return 1.0 if comps else 0.0
    return _clamp01(1.0 - collision_penalty(comps) - spacing_penalty(comps, spacing_limit))


def visual_score(components, expected_count, weights=DEFAULT_WEIGHTS,
                 aspect_limit=ASPECT_LIMIT, spacing_limit=SPACING_LIMIT):
    """The full CADFusion visual grade for a rendered object.

    Returns a dict with the three sub-scores (each in [0, 1]), the weighted
    ``combined`` value in [0, 1], and the final ``score`` in [0, 10] (the harsh
    grader's number out of 10). ``weights`` are (quality, quantity, distribution)
    and must be non-negative with a positive sum; they are renormalised to sum 1.
    """
    comps = list(components)
    w = tuple(float(x) for x in weights)
    if len(w) != 3 or any(x < 0.0 for x in w):
        raise ValueError("weights must be three non-negative numbers")
    wsum = sum(w)
    if wsum <= 0.0:
        raise ValueError("weights must have a positive sum")
    w = tuple(x / wsum for x in w)

    quality = shape_quality_score(comps, aspect_limit)
    quantity = shape_quantity_score(len(comps), expected_count)
    distribution = distribution_score(comps, spacing_limit)
    combined = w[0] * quality + w[1] * quantity + w[2] * distribution
    return {
        "shape_quality": quality,
        "shape_quantity": quantity,
        "distribution": distribution,
        "combined": combined,
        "score": MAX_SCORE * combined,
    }
