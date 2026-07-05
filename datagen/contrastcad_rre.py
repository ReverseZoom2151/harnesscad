"""ContrastCAD "Random Replace and Extrude" (RRE) data augmentation.

Jung, Kim & Kim, *ContrastCAD: Contrastive Learning-Based Representation
Learning for Computer-Aided Design Models* (2024), Section 4.1.

The DeepCAD training corpus is **imbalanced**: 78.38 % of construction sequences
contain a line command but only 19.76 % contain an arc, and 92.60 % of extrusions
are one-sided while symmetric / two-sided together account for < 11 % (paper
Tables 2-3). A model trained on that skew reconstructs arcs and non-one-sided
extrusions poorly. RRE rebalances the data by *synthesising* the rare commands
from the abundant ones, and unlike prior swap-only augmentation it applies to
**every** construction sequence regardless of how many sketch/extrude pairs it
has. Three deterministic operations (paper Section 4.1):

1. **Random Replace (line -> arc).** A portion of the line commands is chosen at
   random and rewritten as arc commands. The endpoint ``(x, y)`` is preserved so
   the profile still closes; the sweep angle ``theta`` is drawn uniformly from the
   integers ``[1, 255]`` and the counter-clockwise flag ``c`` from ``{0, 1}``.
   This turns straight edges into curves, teaching the scarce arc command.

2. **Random Extrude (parameter resampling).** For each extrusion command the
   extrude *type* ``w`` is resampled from ``{0, 1, 2}`` (one-sided / symmetric /
   two-sided) and the two extrude distances ``delta1, delta2`` from the integers
   ``[0, 255]``. This broadens the shape diversity produced by extrusion.

3. **Random pair swap.** Some sketch-and-extrude pairs of the current sequence are
   swapped with pairs drawn from another sequence in the dataset (Wu et al.'s
   augmentation, generalised here to arbitrary pair counts).

A construction sequence is represented paper-faithfully as a list of command
dicts with **8-bit quantised integer** parameters (DeepCAD's 256-level scheme,
Table 1). This module is the deterministic data operation only; the contrastive
encoder is learned and out of scope (see ``bench/contrastcad_contrastive.py`` for
the deterministic loss maths).

Determinism: all randomness is driven by a caller-supplied ``random.Random`` (or
integer seed); the same (sequence, seed) yields byte-identical output. Stdlib
only, no wall clock.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

# --- command vocabulary (DeepCAD / ContrastCAD Table 1) --------------------
SOL = "SOL"
EOS = "EOS"
LINE = "L"
ARC = "A"
CIRCLE = "C"
EXTRUDE = "E"

# Quantisation range of every continuous parameter (8-bit, 256 levels).
QUANT_MIN = 0
QUANT_MAX = 255

# Extrude-type values for parameter ``w``.
ONE_SIDED = 0
SYMMETRIC = 1
TWO_SIDED = 2
EXTRUDE_TYPES: Tuple[int, ...] = (ONE_SIDED, SYMMETRIC, TWO_SIDED)

Command = Dict[str, object]


def _is_type(cmd: Command, kind: str) -> bool:
    return cmd.get("type") == kind


def _rng(seed) -> random.Random:
    if isinstance(seed, random.Random):
        return seed
    return random.Random(seed)


# --- operation 1: random replace (line -> arc) -----------------------------
def replace_lines_with_arcs(sequence: Sequence[Command], seed,
                            replace_prob: float = 0.5) -> List[Command]:
    """Replace a random portion of line commands with arc commands.

    Each ``L`` command is independently replaced with probability ``replace_prob``.
    The replacement arc keeps the line's endpoint ``(x, y)``, samples ``theta`` from
    the integers ``[1, 255]`` and ``c`` from ``{0, 1}`` (paper Section 4.1). Non-line
    commands pass through unchanged. Deterministic given ``seed``.
    """
    if not 0.0 <= replace_prob <= 1.0:
        raise ValueError("replace_prob must be in [0, 1]")
    rng = _rng(seed)
    out: List[Command] = []
    for cmd in sequence:
        if _is_type(cmd, LINE) and rng.random() < replace_prob:
            out.append({
                "type": ARC,
                "x": int(cmd["x"]),
                "y": int(cmd["y"]),
                "theta": rng.randint(1, QUANT_MAX),
                "c": rng.randint(0, 1),
            })
        else:
            out.append(dict(cmd))
    return out


# --- operation 2: random extrude (parameter resampling) --------------------
def randomize_extrusions(sequence: Sequence[Command], seed) -> List[Command]:
    """Resample every extrusion command's type ``w`` and distances ``delta1/2``.

    ``w`` is drawn from ``{0, 1, 2}`` and ``delta1, delta2`` from ``[0, 255]``
    (paper Section 4.1). Other extrusion parameters (orientation, origin, scale,
    boolean) are preserved. Deterministic given ``seed``.
    """
    rng = _rng(seed)
    out: List[Command] = []
    for cmd in sequence:
        if _is_type(cmd, EXTRUDE):
            new = dict(cmd)
            new["w"] = rng.choice(EXTRUDE_TYPES)
            new["delta1"] = rng.randint(QUANT_MIN, QUANT_MAX)
            new["delta2"] = rng.randint(QUANT_MIN, QUANT_MAX)
            out.append(new)
        else:
            out.append(dict(cmd))
    return out


# --- operation 3: random sketch/extrude pair swap --------------------------
def split_pairs(sequence: Sequence[Command]) -> List[List[Command]]:
    """Split a construction sequence into sketch-and-extrude pairs.

    A pair is the run of commands ending at (and including) an extrusion command
    ``E`` -- it may contain several ``SOL`` loops. Trailing commands after the last
    extrusion (e.g. ``EOS``) form a final residual group with no extrusion.
    """
    pairs: List[List[Command]] = []
    current: List[Command] = []
    for cmd in sequence:
        current.append(dict(cmd))
        if _is_type(cmd, EXTRUDE):
            pairs.append(current)
            current = []
    if current:
        pairs.append(current)
    return pairs


def _extrude_pairs(pairs: List[List[Command]]) -> List[int]:
    return [i for i, p in enumerate(pairs) if p and _is_type(p[-1], EXTRUDE)]


def swap_pairs(sequence: Sequence[Command], other: Sequence[Command], seed,
               swap_prob: float = 0.5) -> List[Command]:
    """Swap a random subset of sketch/extrude pairs between two sequences.

    Extrusion-terminated pairs of ``sequence`` are matched positionally against
    those of ``other``; each matched position is swapped with probability
    ``swap_prob``. A residual (non-extrusion) tail is left in place. Returns the
    augmented ``sequence``. Deterministic given ``seed``.
    """
    if not 0.0 <= swap_prob <= 1.0:
        raise ValueError("swap_prob must be in [0, 1]")
    rng = _rng(seed)
    mine = split_pairs(sequence)
    theirs = split_pairs(other)
    my_idx = _extrude_pairs(mine)
    their_idx = _extrude_pairs(theirs)
    for a, b in zip(my_idx, their_idx):
        if rng.random() < swap_prob:
            mine[a] = [dict(c) for c in theirs[b]]
    return [dict(c) for group in mine for c in group]


# --- full RRE pipeline ------------------------------------------------------
def rre_augment(sequence: Sequence[Command], seed,
                other: Optional[Sequence[Command]] = None,
                replace_prob: float = 0.5,
                swap_prob: float = 0.5) -> List[Command]:
    """Apply the full RRE augmentation to one construction sequence.

    Order (paper Section 4.1): line->arc replace, extrude resample, then (if a
    donor ``other`` sequence is supplied) a random sketch/extrude pair swap. A
    single seed drives all three stages via independent sub-streams, so the same
    ``(sequence, other, seed)`` is byte-identical.
    """
    rng = _rng(seed)
    stage = replace_lines_with_arcs(sequence, rng.randint(0, 2 ** 31 - 1),
                                    replace_prob)
    stage = randomize_extrusions(stage, rng.randint(0, 2 ** 31 - 1))
    if other is not None:
        stage = swap_pairs(stage, other, rng.randint(0, 2 ** 31 - 1), swap_prob)
    return stage
