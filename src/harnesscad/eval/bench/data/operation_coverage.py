"""Operation-coverage, diversity and readability metrics for CAD sequence datasets (Zero-to-CAD).

Mined from *Zero-to-CAD: Agentic Synthesis of Interpretable CAD Programs at
Million-Scale Without Real Data*. Zero-to-CAD's argument (Table 1) is that prior
sequence datasets are narrow -- mostly sketch-and-extrude -- whereas a good dataset
must be *replayable*, *readable*, and cover a *broad operation vocabulary*
(Booleans, fillets, chamfers, lofts, sweeps, shells). Those are deterministic,
measurable dataset properties:

*   :func:`operation_coverage` -- fraction of a target operation vocabulary that a
    dataset exercises (beyond sketch-extrude);
*   :func:`operation_diversity` -- normalised Shannon entropy of operation usage
    (1.0 == perfectly uniform coverage, 0.0 == a single operation);
*   :func:`beyond_sketch_extrude` -- fraction of operation instances that are NOT
    plain sketch/extrude; and
*   :func:`readability` -- fraction of sequences whose parameters are *named*
    (Zero-to-CAD's "readable and editable" criterion: explicit named parameters).

Deterministic, stdlib-only.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

__all__ = [
    "RICH_OPERATIONS",
    "operation_coverage",
    "operation_diversity",
    "beyond_sketch_extrude",
    "readability",
]

#: A representative "rich" operation vocabulary (paper Table 1 notes).
RICH_OPERATIONS: tuple = (
    "sketch", "extrude", "revolve", "sweep", "loft", "shell",
    "fillet", "chamfer", "boolean", "hole", "pattern", "draft",
)

_SKETCH_EXTRUDE = {"sketch", "extrude"}


def _flatten(sequences: Sequence[Sequence[str]]) -> list:
    return [op.lower() for seq in sequences for op in seq]


def operation_coverage(
    sequences: Sequence[Sequence[str]],
    vocabulary: Sequence[str] = RICH_OPERATIONS,
) -> float:
    """Fraction of the target operation vocabulary that appears at least once."""
    if not vocabulary:
        raise ValueError("vocabulary must be non-empty")
    used = set(_flatten(sequences))
    vocab = {v.lower() for v in vocabulary}
    return len(used & vocab) / len(vocab)


def operation_diversity(sequences: Sequence[Sequence[str]]) -> float:
    """Normalised Shannon entropy of operation usage, in ``[0, 1]``.

    1.0 means every distinct operation is used equally often; 0.0 means a single
    operation dominates entirely. An empty dataset raises.
    """
    ops = _flatten(sequences)
    if not ops:
        raise ValueError("no operations to measure")
    counts: Dict[str, int] = {}
    for op in ops:
        counts[op] = counts.get(op, 0) + 1
    n = len(ops)
    distinct = len(counts)
    if distinct <= 1:
        return 0.0
    entropy = -sum((c / n) * math.log(c / n) for c in counts.values())
    return entropy / math.log(distinct)


def beyond_sketch_extrude(sequences: Sequence[Sequence[str]]) -> float:
    """Fraction of operation instances that are NOT plain sketch/extrude."""
    ops = _flatten(sequences)
    if not ops:
        raise ValueError("no operations to measure")
    rich = sum(1 for op in ops if op not in _SKETCH_EXTRUDE)
    return rich / len(ops)


def readability(named_flags: Sequence[bool]) -> float:
    """Fraction of sequences with named (readable/editable) parameters."""
    if not named_flags:
        raise ValueError("need at least one sequence")
    return sum(1 for f in named_flags if f) / len(named_flags)
