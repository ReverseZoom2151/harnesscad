"""MUSE geometry-issue -> flag classifier (Stage-2 tri-state semantics).

Reproduces the exact deterministic classification in
``src/judge_system/geometry_metrics.py`` of the muse-benchmark repo (Dong et
al., "MUSE") that turns a raw list of OCCT validator issues into the Stage-2
geometry flags -- *without* the OCCT dependency. The validator subprocess is
external; the classification of its ``issues`` payload is pure logic and is what
this module isolates.

Each issue is a dict ``{"issue_type": str, "severity": "error"|"warning"|...}``.
The classifier applies the repo's ``no_error`` rule per flag:

  * if the code did not run (``code_valid`` is False)  -> flag is None;
  * else if no issue of the relevant type(s) is present -> True;
  * else True iff none of those issues has ``severity == "error"`` (warnings do
    not fail the flag).

DIVERGENCE from ``bench/muse_scorecard`` -- this module makes precise a
definitional gap the funnel scorecard glosses over:

  * The repo's ``watertight`` flag is COMBINED: True iff there is no
    ``Watertightness`` *and* no ``NonManifoldEdge`` error. So in the repo
    ``watertight`` already IMPLIES ``manifold``; they are not independent.
    ``muse_scorecard`` instead treats ``watertight`` and ``manifold`` as two
    independent binary checks and ANDs all four -- double-counting the
    non-manifold condition.
  * The repo additionally exposes ``watertight_strict`` (no ``Watertightness``
    error only, ignoring manifoldness), which the funnel omits entirely.
  * The repo's Stage-2 also tracks ``normal_consistency``, ``volume_valid``,
    ``bbox_valid`` and ``occt_valid``; the funnel's four-check ``geom_valid``
    uses only watertight/manifold/self-intersection/overlap.
  * ``overlap_free`` in the funnel corresponds to a SEPARATE interpenetration
    subprocess (``evaluate_interpenetration``, rel-volume threshold 0.01 of the
    smaller solid), not to any validator issue type -- see
    ``muse2_interpenetration_ratio``.

This module is the faithful re-encoding; use it to derive the funnel's
``watertight``/``manifold``/``self_intersection_free`` inputs consistently.

No wall clock, no randomness.
"""

from __future__ import annotations

# Issue-type groupings, taken verbatim from geometry_metrics.evaluate_geometry.
_WATERTIGHT_TYPES = ("Watertightness", "NonManifoldEdge")   # combined
_WATERTIGHT_STRICT_TYPES = ("Watertightness",)
_MANIFOLD_TYPES = ("NonManifoldEdge",)
_SELF_INTERSECTION_TYPES = ("SelfIntersection",)
_NORMAL_TYPES = ("NormalConsistency",)
_VOLUME_TYPES = ("ZeroVolume", "NegativeVolume")
_BBOX_TYPES = ("DegenerateBBox", "InfiniteBBox", "BoundingBox")
_OCCT_TYPES = ("OCCTValidity",)


def _no_error(issues, names, code_valid):
    """Repo ``no_error``: None if code invalid; True if no error of ``names``."""
    if not code_valid:
        return None
    related = [i for i in issues if i.get("issue_type") in names]
    if not related:
        return True
    return not any(i.get("severity") == "error" for i in related)


def _count(issues, name):
    return sum(1 for i in issues
               if i.get("issue_type") == name and i.get("severity") == "error")


def classify_geometry_issues(issues, code_valid=True):
    """Classify a validator issue list into the MUSE Stage-2 flag set.

    issues : iterable of {"issue_type", "severity", ...}.
    code_valid : whether the candidate code executed (False forces every
        tri-state flag to None, matching the repo).

    Returns a dict with tri-state flags (True/False/None):
      watertight, watertight_strict, manifold, self_intersection_free,
      normal_consistency, volume_valid, bbox_valid, occt_valid
    plus integer error counts:
      watertight_error_count, self_intersection_error_count,
      non_manifold_error_count, volume_error_count, bbox_error_count.
    """
    issues = list(issues)
    return {
        "watertight": _no_error(issues, _WATERTIGHT_TYPES, code_valid),
        "watertight_strict": _no_error(issues, _WATERTIGHT_STRICT_TYPES, code_valid),
        "manifold": _no_error(issues, _MANIFOLD_TYPES, code_valid),
        "self_intersection_free": _no_error(issues, _SELF_INTERSECTION_TYPES, code_valid),
        "normal_consistency": _no_error(issues, _NORMAL_TYPES, code_valid),
        "volume_valid": _no_error(issues, _VOLUME_TYPES, code_valid),
        "bbox_valid": _no_error(issues, _BBOX_TYPES, code_valid),
        "occt_valid": _no_error(issues, _OCCT_TYPES, code_valid),
        "watertight_error_count": _count(issues, "Watertightness"),
        "self_intersection_error_count": _count(issues, "SelfIntersection"),
        "non_manifold_error_count": _count(issues, "NonManifoldEdge"),
        "volume_error_count": _count(issues, "ZeroVolume") + _count(issues, "NegativeVolume"),
        "bbox_error_count": _count(issues, "BoundingBox"),
    }


def geometry_valid(flags):
    """Repo-style Stage-2 verdict from a classified flag dict.

    True iff watertight, self_intersection_free, volume_valid and bbox_valid are
    all True (manifold is subsumed by the combined ``watertight``). Any None or
    False -> False. This is the repo's stricter geometry gate, wider than the
    funnel's four-check version.
    """
    required = ("watertight", "self_intersection_free", "volume_valid", "bbox_valid")
    return all(flags.get(k) is True for k in required)


def to_funnel_geometry(flags):
    """Project the repo flags onto the funnel's four binary geometry checks.

    Returns {watertight, manifold, self_intersection_free} as 0/1 (None->0),
    ready to merge with an externally-computed ``overlap_free`` before feeding
    ``bench.muse_scorecard.evaluate_design``. Note ``watertight`` here is the
    repo's combined flag, so passing both it and ``manifold`` to the funnel
    double-counts non-manifoldness -- prefer ``watertight_strict`` for the
    funnel's ``watertight`` slot if independence is intended.
    """
    return {
        "watertight": 1 if flags.get("watertight") is True else 0,
        "manifold": 1 if flags.get("manifold") is True else 0,
        "self_intersection_free": 1 if flags.get("self_intersection_free") is True else 0,
    }
