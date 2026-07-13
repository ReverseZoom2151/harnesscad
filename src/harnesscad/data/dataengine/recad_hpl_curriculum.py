"""ReCAD Hierarchical Primitive Learning curriculum (ReCAD, AAAI 2026).

ReCAD trains under a curriculum-learning strategy called **Hierarchical
Primitive Learning (HPL)** that follows the inherent structure of sketch-extrude
CAD models.  The five hierarchical primitives (Eq. 1) form a
composition hierarchy::

    P = { L (loop), F (face), S (sketch), SE (sketch-extrude), MSE (multi-SE) }

The curriculum introduces primitives in this *simple-to-complex* order: a loop
is a single closed path; a face groups loops; a sketch groups faces on a plane;
an SE pair adds an extrusion; an MSE composes multiple SE pairs into a complete
model.  Each stage "builds upon the structure of the preceding one".

Within each primitive level, ReCAD further orders training samples by
**difficulty, defined as the number of curves involved**, learning simple
(few-curve) examples before richer ones.

This module implements the deterministic curriculum *ordering / scheduling*: it
sorts a corpus by (primitive-level rank, curve count) and can emit stage-by-stage
batches.  It is distinct from ``dataengine.creft_rewards`` /
``creft_data_engine`` (a curriculum over three attribute-recovery *tasks* with
scalar difficulty weights) and from ``dataengine.modality_schedule`` (an
input-modality schedule): HPL orders by the CAD *composition hierarchy*.  Pure
stdlib, deterministic.
"""

from __future__ import annotations

from typing import Iterable, Sequence

# The five hierarchical primitives in curriculum (simple -> complex) order.
PRIMITIVE_ORDER = ("L", "F", "S", "SE", "MSE")
_RANK = {name: i for i, name in enumerate(PRIMITIVE_ORDER)}


def primitive_rank(primitive: str) -> int:
    """Curriculum rank of a primitive level (0 = ``L`` .. 4 = ``MSE``)."""
    key = str(primitive)
    if key not in _RANK:
        raise ValueError(
            f"unknown primitive {primitive!r}; expected one of {PRIMITIVE_ORDER}")
    return _RANK[key]


def difficulty_key(sample) -> tuple:
    """Sort key ``(primitive_rank, curve_count)`` for a curriculum sample.

    ``sample`` is a mapping with keys ``"primitive"`` and ``"curves"`` (the
    number of curves, the paper's within-level difficulty measure).
    """
    curves = int(sample["curves"])
    if curves < 0:
        raise ValueError("curve count must be non-negative")
    return (primitive_rank(sample["primitive"]), curves)


def order_curriculum(samples: Iterable) -> list:
    """Return samples ordered simple-to-complex by (primitive level, curves).

    The sort is *stable*, so samples sharing a (level, curve-count) key keep
    their original relative order -- the schedule is fully deterministic.
    """
    return sorted(samples, key=difficulty_key)


def stage_batches(samples: Iterable) -> list:
    """Group the ordered curriculum into per-primitive-level stages.

    Returns a list of ``(primitive, [samples...])`` tuples in curriculum order,
    each stage internally ordered by curve count.  Empty levels are omitted.
    """
    ordered = order_curriculum(samples)
    stages = []
    for sample in ordered:
        prim = str(sample["primitive"])
        if not stages or stages[-1][0] != prim:
            stages.append((prim, []))
        stages[-1][1].append(sample)
    return stages


def curriculum_indices(samples: Sequence) -> list:
    """Return original indices of ``samples`` in curriculum-presentation order."""
    return [i for i, _ in sorted(
        enumerate(samples), key=lambda pair: difficulty_key(pair[1]))]
