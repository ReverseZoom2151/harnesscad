"""Validity check and physics-aware rollback for brick generation (BRICKGPT).

Paper: "Generating Physically Stable and Buildable Brick Structures from Text",
Section 4.2 and Algorithm 1. During autoregressive inference BRICKGPT enforces
feasibility with two mechanisms:

1. **Brick-by-brick validity check / rejection sampling.** Each newly predicted
   brick must be well-formatted (present in the brick library), lie inside the
   workspace, and not collide with the existing structure
   (``V_t intersect V_i = empty``). Invalid bricks are rejected and resampled.

2. **Physics-aware rollback.** After a candidate design is complete, its
   stability score ``S`` is computed. If unstable, the design is rolled back to
   the state *before the first unstable brick* was generated:
   ``B' = [b_1, ..., b_{min(I)-1}]`` where ``I`` is the set of indices of
   unstable bricks (Algorithm 1, lines 11-15), and generation resumes from
   there.

This module implements the *deterministic* pieces of that inference loop
(stdlib only). The learned autoregressive model (LLaMA-3.2 fine-tune) that
proposes candidate bricks is research-heavy/external; here a candidate brick is
supplied by the caller (e.g. an iterator of proposals), and this module performs
the validity gating and the physics-aware rollback around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from geometry.brick_structure import (
    STANDARD_BRICKS,
    Brick,
    BrickStructure,
    bricks_overlap,
)
from verifiers.brick_stability import (
    DEFAULT_FRICTION_CAPACITY,
    analyze_stability,
)


# ---------------------------------------------------------------------------
# Per-brick validity check (rejection-sampling gate).
# ---------------------------------------------------------------------------


def is_valid_brick(
    brick: Brick,
    grid_h: int = 20,
    grid_w: int = 20,
    grid_d: int = 20,
    library: Iterable[tuple[int, int]] = STANDARD_BRICKS,
) -> bool:
    """True if ``brick`` is well-formatted: in the library and inside the grid."""
    return brick.in_library(library) and brick.in_bounds(grid_h, grid_w, grid_d)


def is_valid_placement(
    existing: Sequence[Brick],
    brick: Brick,
    grid_h: int = 20,
    grid_w: int = 20,
    grid_d: int = 20,
    library: Iterable[tuple[int, int]] = STANDARD_BRICKS,
) -> bool:
    """True if ``brick`` is a valid, non-colliding addition to ``existing``.

    Implements the inference-time validity check of Section 4.2: well-formatted,
    in-bounds, and ``V_t intersect V_i = empty`` for all placed bricks.
    """
    if not is_valid_brick(brick, grid_h, grid_w, grid_d, library):
        return False
    return not any(bricks_overlap(brick, other) for other in existing)


def rejection_sample(
    existing: Sequence[Brick],
    candidates: Iterable[Brick],
    grid_h: int = 20,
    grid_w: int = 20,
    grid_d: int = 20,
    library: Iterable[tuple[int, int]] = STANDARD_BRICKS,
) -> Optional[Brick]:
    """Return the first candidate that passes the validity check, or ``None``.

    Deterministic model of brick-by-brick rejection sampling (Algorithm 1,
    lines 3-7): iterate proposals in order and accept the first valid one.
    """
    for cand in candidates:
        if is_valid_placement(existing, cand, grid_h, grid_w, grid_d, library):
            return cand
    return None


# ---------------------------------------------------------------------------
# Physics-aware rollback (Algorithm 1, lines 11-15).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RollbackResult:
    structure: BrickStructure
    removed: int  # number of bricks removed by rollback
    rollbacks: int  # number of rollback iterations performed
    stable: bool


def first_unstable_index(
    structure: BrickStructure,
    friction_capacity: float = DEFAULT_FRICTION_CAPACITY,
) -> Optional[int]:
    """Index of the first unstable brick (``min I``), or ``None`` if all stable."""
    result = analyze_stability(structure, friction_capacity)
    unstable = result.unstable_indices()
    return min(unstable) if unstable else None


def physics_aware_rollback(
    structure: BrickStructure,
    friction_capacity: float = DEFAULT_FRICTION_CAPACITY,
    max_rollbacks: int = 100,
) -> RollbackResult:
    """Roll a design back to its last stable prefix (Algorithm 1, lines 11-15).

    While the structure is unstable, find the first unstable brick ``i = min I``
    and truncate the design to ``[b_1, ..., b_{i-1}]``. Repeat (each truncation
    can expose newly-unstable bricks) until stable or ``max_rollbacks`` is
    exceeded.
    """
    current = structure
    rollbacks = 0
    original_len = len(structure.bricks)
    while rollbacks < max_rollbacks:
        idx = first_unstable_index(current, friction_capacity)
        if idx is None:
            return RollbackResult(
                current, original_len - len(current.bricks), rollbacks, True
            )
        current = current.prefix(idx)
        rollbacks += 1
    # Exceeded rollback budget; report current (possibly still unstable) state.
    stable = first_unstable_index(current, friction_capacity) is None
    return RollbackResult(
        current, original_len - len(current.bricks), rollbacks, stable
    )


def build_with_validity_and_rollback(
    candidates: Sequence[Brick],
    grid_h: int = 20,
    grid_w: int = 20,
    grid_d: int = 20,
    library: Iterable[tuple[int, int]] = STANDARD_BRICKS,
    friction_capacity: float = DEFAULT_FRICTION_CAPACITY,
    max_rollbacks: int = 100,
) -> RollbackResult:
    """Full deterministic inference loop over an ordered list of candidate bricks.

    Adds each candidate that passes the validity check (skipping invalid /
    colliding ones, the rejection-sampling effect), then applies physics-aware
    rollback so the returned structure is stable (or the best stable prefix
    within the rollback budget).
    """
    placed: list[Brick] = []
    for cand in candidates:
        if is_valid_placement(placed, cand, grid_h, grid_w, grid_d, library):
            placed.append(cand)
    structure = BrickStructure(tuple(placed), grid_h, grid_w, grid_d)
    return physics_aware_rollback(structure, friction_capacity, max_rollbacks)
