"""Feedback- and error-type taxonomies for CADCodeVerify (paper App. C).

Deterministic re-implementation of the two qualitative-analysis taxonomies from
"Generating CAD Code with Vision-Language Models for 3D Designs" (Alrashedy et
al., ICLR 2025):

  * Feedback types (App. C.1): every corrective feedback string is one of
    Structural / Dimensional / Positional feedback.
  * Error types (App. C.2, following Yuan et al. 2024): Structural
    Configuration, Spatial Precision, Logical, Correct, Failure.

The paper obtains these labels via human annotators; here we provide (a) the
taxonomy definitions, (b) a deterministic keyword classifier that assigns a
feedback string to a feedback type, and (c) distribution reporting over a set
of labels (the pie/bar charts of Figures 7-8).  The keyword classifier is a
transparent heuristic, not a learned model.
"""
from __future__ import annotations

# ---- Feedback types (App. C.1) ------------------------------------------

FEEDBACK_TYPES = {
    "structural": "correct the structure of the object (e.g. make cylindrical, "
                  "adjust corner shape)",
    "dimensional": "instructions about size and scale (e.g. increase height, "
                   "reduce width)",
    "positional": "focus on alignment of objects (e.g. center object, align "
                  "with base)",
}

_DIMENSIONAL_WORDS = (
    "height", "width", "depth", "length", "size", "scale", "thickness",
    "diameter", "radius", "taller", "shorter", "wider", "narrower", "bigger",
    "smaller", "larger", "increase", "decrease", "enlarge", "shrink", "dimension",
)
_POSITIONAL_WORDS = (
    "center", "centre", "align", "alignment", "position", "move", "shift",
    "offset", "place", "relocate", "base", "corner position", "top of", "bottom of",
)
_STRUCTURAL_WORDS = (
    "cylindrical", "cylinder", "shape", "structure", "corner", "round", "curve",
    "surface", "hole", "edge", "fillet", "chamfer", "add", "remove", "connect",
    "attach", "merge", "hollow", "solid",
)


def classify_feedback(text):
    """Assign a feedback string to a feedback type by keyword voting.

    Ties and no-match default to "structural" (the dominant category in the
    paper, App. C.1).  Deterministic and case-insensitive.
    """
    low = str(text).lower()
    scores = {
        "dimensional": sum(low.count(w) for w in _DIMENSIONAL_WORDS),
        "positional": sum(low.count(w) for w in _POSITIONAL_WORDS),
        "structural": sum(low.count(w) for w in _STRUCTURAL_WORDS),
    }
    best = max(scores.values())
    if best == 0:
        return "structural"
    # Priority order on ties: dimensional, positional, structural.
    for key in ("dimensional", "positional", "structural"):
        if scores[key] == best:
            return key
    return "structural"  # pragma: no cover


def feedback_distribution(texts):
    """Count + fraction of feedback types over a set of feedback strings."""
    labels = [classify_feedback(t) for t in texts]
    return _distribution(labels, FEEDBACK_TYPES.keys())


# ---- Error types (App. C.2) ---------------------------------------------

ERROR_TYPES = {
    "structural_configuration": "the structure of the object is incorrectly "
                                "arranged",
    "spatial_precision": "a minor error in spatial parameters (height, width, "
                         "volume)",
    "logical": "implausible configuration that does not resemble real-world "
               "contexts",
    "correct": "object without errors",
    "failure": "object failed to generate due to a compile error",
}


def normalize_error_type(label):
    key = str(label).strip().lower().replace(" ", "_")
    aliases = {
        "structural": "structural_configuration",
        "structural_config": "structural_configuration",
        "spatial": "spatial_precision",
        "failure_rate": "failure",
        "failed": "failure",
        "ok": "correct",
    }
    key = aliases.get(key, key)
    if key not in ERROR_TYPES:
        raise ValueError("unknown error type: %r" % (label,))
    return key


def error_distribution(labels):
    """Count + fraction of error types (Figure 8 pie chart)."""
    normed = [normalize_error_type(l) for l in labels]
    return _distribution(normed, ERROR_TYPES.keys())


def majority_vote(annotations):
    """Resolve a per-object error label from several annotators by majority.

    ``annotations`` is a list of label strings; ties are broken by taxonomy
    order (structural_configuration first) for determinism.
    """
    normed = [normalize_error_type(a) for a in annotations]
    if not normed:
        raise ValueError("no annotations")
    counts = {}
    for a in normed:
        counts[a] = counts.get(a, 0) + 1
    best = max(counts.values())
    for key in ERROR_TYPES:
        if counts.get(key, 0) == best:
            return key
    return normed[0]  # pragma: no cover


def _distribution(labels, keys):
    total = len(labels)
    if total == 0:
        raise ValueError("no labels")
    counts = {k: 0 for k in keys}
    for l in labels:
        counts[l] += 1
    return {
        "total": total,
        "counts": counts,
        "fractions": {k: counts[k] / total for k in counts},
    }
