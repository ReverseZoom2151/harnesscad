"""CAD data-representation completeness taxonomy.

The thesis is that in AI+CAD the *data representation format* -- not the
network -- sets the ceiling on what can be learned, and that different formats
retain different engineering semantics. It contrasts a flat sketch-extrude
"solid modeling" representation against a three-level "industrial-level
parametric feature" architecture, and argues completeness is a function of which
semantic layers survive the encoding.

This module turns that qualitative argument into a deterministic checklist score.
Each representation is described by which of the canonical engineering-semantic
layers it preserves; the module ranks representations, reports the missing layers
that cap their learnability, and assigns a three-tier level
(solid-modeling vs. feature-modeling vs. industrial-parametric).

No model, no geometry -- a pure schema/scoring utility providing a
table-style comparison.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

__all__ = [
    "SEMANTIC_LAYERS",
    "LEVEL_REQUIREMENTS",
    "score_representation",
    "compare_representations",
]

# Canonical engineering-semantic layers, ordered from geometry-only to
# industrial parametric feature modeling.
SEMANTIC_LAYERS: Sequence[str] = (
    "geometry",            # points / mesh / raw solid
    "sketch",              # 2D parametric sketches
    "constraints",         # sketch/assembly constraints
    "feature_tree",        # ordered feature operations (extrude, fillet, ...)
    "modeling_order",      # design-history sequence
    "topological_naming",  # persistent face/edge references + selection
)

# The three tiers, each requiring a cumulative layer set.
LEVEL_REQUIREMENTS: Mapping[str, Sequence[str]] = {
    "solid_modeling": ("geometry", "sketch"),
    "feature_modeling": ("geometry", "sketch", "feature_tree", "modeling_order"),
    "industrial_parametric": (
        "geometry",
        "sketch",
        "constraints",
        "feature_tree",
        "modeling_order",
        "topological_naming",
    ),
}

# Level ordering for reporting the best achieved tier.
_LEVEL_ORDER = ("none", "solid_modeling", "feature_modeling", "industrial_parametric")


def _normalise(layers: Sequence[str]) -> set:
    unknown = set(layers) - set(SEMANTIC_LAYERS)
    if unknown:
        raise ValueError(f"unknown semantic layers: {sorted(unknown)}")
    return set(layers)


def score_representation(name: str, layers: Sequence[str]) -> Dict[str, object]:
    """Score one representation by the semantic layers it preserves.

    Returns a dict with:
      ``coverage``  : fraction of canonical layers preserved (0..1);
      ``missing``   : ordered list of absent layers (the learnability cap);
      ``level``     : the highest tier fully satisfied (``LEVEL_REQUIREMENTS``);
      ``preserved`` : ordered list of preserved layers.
    """
    have = _normalise(layers)
    preserved = [l for l in SEMANTIC_LAYERS if l in have]
    missing = [l for l in SEMANTIC_LAYERS if l not in have]
    level = "none"
    for tier, req in LEVEL_REQUIREMENTS.items():
        if have.issuperset(req):
            level = tier
    return {
        "name": name,
        "coverage": len(preserved) / len(SEMANTIC_LAYERS),
        "preserved": preserved,
        "missing": missing,
        "level": level,
    }


def compare_representations(
    reps: Mapping[str, Sequence[str]]
) -> List[Dict[str, object]]:
    """Rank several representations best-first.

    Ordering is by achieved tier, then coverage, then name (deterministic). This
    reproduces the layered-vs-flat comparison: a representation carrying
    topological naming + constraints ranks strictly above a bare sketch-extrude
    representation.
    """
    scored = [score_representation(n, l) for n, l in reps.items()]
    scored.sort(
        key=lambda s: (_LEVEL_ORDER.index(s["level"]), s["coverage"], s["name"]),
        reverse=True,
    )
    return scored
