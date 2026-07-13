"""CADSmith benchmark difficulty tiers (CADSmith sec. III-A).

The CADSmith benchmark stratifies prompts into three difficulty tiers by the
kind of geometry and the number of CadQuery operations required:

  * T1 - Basic Primitives:   single geometric shapes (boxes, cylinders, cones,
                             tori, prisms, domes); 1-3 operations.
  * T2 - Engineering Parts:  parts requiring boolean operations (brackets,
                             flanges, gears, shafts, hole patterns, counterbored
                             fasteners); 3-8 operations.
  * T3 - Complex Parts:      multi-feature parts requiring workplane changes,
                             lofts, sweeps, shells, revolves, multi-body unions;
                             5-15 operations.

This module encodes that taxonomy deterministically: it classifies a prompt/
op-list into a tier from the operation count and the operation vocabulary used,
and exposes the tier definitions (op-count band, representative operations,
example parts). The tiers overlap in op-count (T2 3-8, T3 5-15), so operation
*kind* breaks ties — presence of any complex operation (loft/sweep/shell/
revolve/workplane-change/multi-body) forces T3, presence of a boolean forces at
least T2.

Stdlib only, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# Operation vocabulary
# --------------------------------------------------------------------------- #
# Operations that, if present, mark a part as at least "engineering" (T2).
BOOLEAN_OPS = frozenset({"cut", "union", "fuse", "intersect", "common",
                         "hole", "cboreHole", "cboied", "cskHole", "cbore"})

# Operations that mark a part as "complex" (T3).
COMPLEX_OPS = frozenset({"loft", "sweep", "shell", "revolve", "revolved",
                         "workplane_change", "multi_body", "twistExtrude"})

PRIMITIVE_OPS = frozenset({"box", "cylinder", "cone", "torus", "prism", "dome",
                           "sphere", "rect", "circle", "extrude"})


# --------------------------------------------------------------------------- #
# Tier definitions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TierSpec:
    tier: str
    name: str
    op_min: int
    op_max: int
    representative_ops: Tuple[str, ...]
    examples: Tuple[str, ...]

    def op_count_in_band(self, n: int) -> bool:
        return self.op_min <= n <= self.op_max


T1 = TierSpec(
    "T1", "Basic Primitives", 1, 3,
    ("box", "cylinder", "cone", "torus", "prism", "dome"),
    ("box", "cylinder", "cone", "torus", "prism", "dome"),
)
T2 = TierSpec(
    "T2", "Engineering Parts", 3, 8,
    ("cut", "union", "hole", "cboreHole"),
    ("bracket", "flange", "gear", "shaft", "plate with hole pattern",
     "counterbored fastener"),
)
T3 = TierSpec(
    "T3", "Complex Parts", 5, 15,
    ("loft", "sweep", "shell", "revolve", "workplane_change", "multi_body"),
    ("multi-feature part", "flanged shaft coupling", "quadcopter frame"),
)

TIERS: Tuple[TierSpec, ...] = (T1, T2, T3)


def tier_spec(tier: str) -> TierSpec:
    for t in TIERS:
        if t.tier == tier:
            return t
    raise KeyError(tier)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TierClassification:
    tier: str
    op_count: int
    reason: str


def classify(ops: Sequence[str]) -> TierClassification:
    """Classify an operation sequence into a difficulty tier.

    Rules (kind dominates count where bands overlap):
      1. Any complex operation  -> T3.
      2. Otherwise any boolean  -> T2 (unless the count alone reaches T3's floor
         with no complex op, in which case it stays T2 — booleans without
         complex ops are the T2 signature).
      3. Otherwise (primitives only) -> T1 if within 1-3 ops, else T2 by count.
    """
    n = len(ops)
    kinds = set(ops)
    has_complex = bool(kinds & COMPLEX_OPS)
    has_boolean = bool(kinds & BOOLEAN_OPS)

    if has_complex:
        return TierClassification("T3", n, "uses complex operation(s)")
    if has_boolean:
        return TierClassification("T2", n, "uses boolean operation(s)")
    if n <= T1.op_max:
        return TierClassification("T1", n, "primitives only within 1-3 ops")
    return TierClassification("T2", n,
                              "primitives only but op count exceeds T1 band")


def op_count_tier(n: int) -> Optional[str]:
    """The tier whose op-count band contains ``n`` with no ambiguity, else None.

    Because T2 (3-8) and T3 (5-15) overlap and T1/T2 share 3, most counts are
    ambiguous by count alone; this returns a tier only for counts that fall in
    exactly one band (used for sanity checks, not primary classification).
    """
    hits = [t.tier for t in TIERS if t.op_count_in_band(n)]
    return hits[0] if len(hits) == 1 else None
