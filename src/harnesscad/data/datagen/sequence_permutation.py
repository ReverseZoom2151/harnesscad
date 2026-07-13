"""Shape-preserving permutation augmentation for ContrastCAD robustness tests.

Jung, Kim & Kim, *ContrastCAD* (2024), Section 5.4.4 ("Results on robustness of
the model with respect to permutation changes").

A single CAD shape can be written with many equivalent construction sequences:
the curve commands inside a closed loop can start at any vertex, and independent
loops can be listed in any order, yet the resulting solid is identical. A good
representation should map these equivalent sequences to nearby latent vectors.
The paper measures that robustness by generating permuted-but-equivalent
sequences for the three most frequent test patterns:

    P1: (SOL, L, L, L, L, E)          -> cyclic shift of the four line commands
    P2: (SOL, L, L, L, L, L, L, E)    -> cyclic shift of the six line commands
    P3: (SOL, C, SOL, C, E)           -> swap the two circle loops

This module generates those equivalences deterministically. ``cyclic_shift_loop``
rotates the ordered curve commands of one loop while preserving orientation (the
paper's "(L2, L3, L4, L1)" style shift); ``swap_circle_loops`` exchanges two
single-circle ``SOL`` loops that share an extrusion (pattern P3); and
``permute_sequence`` applies a random equivalence-preserving permutation to a
whole construction sequence. Because these transforms never touch a coordinate or
parameter value -- only command order within a loop / loop order within a profile
-- the represented 3D shape is unchanged.

Commands use the same quantised-integer dict representation as
``datagen/contrastcad_rre.py``. Determinism via a caller-supplied
``random.Random`` or seed. Stdlib only.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from harnesscad.data.datagen.replace_extrude_augment import (
    ARC,
    CIRCLE,
    EXTRUDE,
    LINE,
    SOL,
    Command,
    split_pairs,
)

# Curve command types that live inside a loop (order can be cyclically shifted).
_CURVE_TYPES = frozenset({LINE, ARC, CIRCLE})


def _rng(seed) -> random.Random:
    if isinstance(seed, random.Random):
        return seed
    return random.Random(seed)


def cyclic_shift_loop(loop: Sequence[Command], shift: int) -> List[Command]:
    """Cyclically shift the curve commands of one ``SOL`` loop by ``shift``.

    ``loop`` is ``[SOL, curve, curve, ...]``. The leading ``SOL`` (and any
    trailing extrusion) stay put; only the ordered curve commands rotate, which
    keeps orientation and therefore the drawn profile identical (paper's
    ``(L1,L2,L3,L4) -> (L2,L3,L4,L1)`` shift). ``shift`` is taken modulo the number
    of curve commands.
    """
    head = [dict(c) for c in loop if c.get("type") == SOL]
    tail = [dict(c) for c in loop if c.get("type") == EXTRUDE]
    curves = [dict(c) for c in loop if c.get("type") in _CURVE_TYPES]
    if curves:
        k = shift % len(curves)
        curves = curves[k:] + curves[:k]
    return head + curves + tail


def _split_loops(pair: Sequence[Command]) -> List[List[Command]]:
    """Split one sketch/extrude pair into its ``SOL`` loops (extrusion excluded)."""
    loops: List[List[Command]] = []
    current: List[Command] = []
    for cmd in pair:
        if cmd.get("type") == EXTRUDE:
            continue
        if cmd.get("type") == SOL:
            if current:
                loops.append(current)
            current = [dict(cmd)]
        elif current:
            current.append(dict(cmd))
    if current:
        loops.append(current)
    return loops


def swap_circle_loops(pair: Sequence[Command]) -> List[Command]:
    """Swap the first two single-circle loops of a sketch/extrude pair (P3).

    Reproduces the paper's P3 permutation ``(SOL, C, SOL, C, E)`` -> circles
    exchanged. The trailing extrusion is preserved. If the pair does not contain
    at least two circle loops it is returned unchanged.
    """
    loops = _split_loops(pair)
    tail = [dict(c) for c in pair if c.get("type") == EXTRUDE]
    circle_positions = [
        i for i, lp in enumerate(loops)
        if len(lp) == 2 and lp[0].get("type") == SOL and lp[1].get("type") == CIRCLE
    ]
    if len(circle_positions) >= 2:
        i, j = circle_positions[0], circle_positions[1]
        loops[i], loops[j] = loops[j], loops[i]
    return [c for lp in loops for c in lp] + tail


def permute_sequence(sequence: Sequence[Command], seed) -> List[Command]:
    """Return a shape-equivalent permuted construction sequence.

    For every sketch/extrude pair: each loop's curve commands are cyclically
    shifted by a random amount, and single-circle loops within the pair are
    randomly reordered. Coordinates and parameters are untouched, so the
    represented shape is identical to the input. Deterministic given ``seed``.
    """
    rng = _rng(seed)
    out: List[Command] = []
    for pair in split_pairs(sequence):
        tail = [dict(c) for c in pair if c.get("type") == EXTRUDE]
        loops = _split_loops(pair)
        # Cyclically shift each loop's curves.
        shifted = [cyclic_shift_loop(lp, rng.randint(0, len(lp)))
                   for lp in loops]
        # Randomly reorder single-circle loops (order-independent within a sketch).
        circle_positions = [
            i for i, lp in enumerate(shifted)
            if len(lp) == 2 and lp[0].get("type") == SOL
            and lp[1].get("type") == CIRCLE
        ]
        if len(circle_positions) >= 2:
            perm = circle_positions[:]
            rng.shuffle(perm)
            reordered = [dict() for _ in shifted]
            for src, dst in zip(circle_positions, perm):
                reordered[dst] = shifted[src]
            for i, lp in enumerate(shifted):
                if i not in circle_positions:
                    reordered[i] = lp
            shifted = reordered
        for lp in shifted:
            out.extend(lp)
        out.extend(tail)
    return out
