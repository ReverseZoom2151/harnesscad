"""HistCAD dataset-quality metrics over history sequences.

Deterministic statistics HistCAD uses to argue its representation is more
compact and more constraint-rich than DeepCAD / Text2CAD (paper Sec. III-D,
Tables II/III/VI). These operate on collections of modeling sequences
(``reconstruction.histcad_sequence.ModelingSequence`` or any object exposing
the same shape) — no learned model:

  * :func:`sequence_length_stats` — mean / median / 95th-percentile token
    length (Table VI style);
  * :func:`constraint_overhead` — relative token increase from including
    constraints (paper reports ~32.7%);
  * :func:`constraint_distribution` — normalised frequency per constraint type
    (Table II) and its total-variation distance from a reference distribution;
  * :func:`flattening_ratio` — how much a flat (deduplicated) primitive set
    shrinks a nested face-loop hierarchy (structural-redundancy removal);
  * :func:`hierarchy_free` — fraction of sequences with no nested (hole) loops.

Stdlib-only, deterministic.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from harnesscad.domain.reconstruction.sequences.histcad_sequence import (
    ModelingSequence, token_estimate, symmetric_difference,
)
from harnesscad.core.state.histcad_constraint_model import CONSTRAINT_TYPES, constraint_histogram

#: HistCAD Table II reference frequencies (fractions summing to ~1).
REFERENCE_DISTRIBUTION: Dict[str, float] = {
    "coincident": 0.2733, "horizontal": 0.2172, "perpendicular": 0.1697,
    "parallel": 0.1624, "vertical": 0.0757, "equal": 0.0411, "tangent": 0.0330,
    "concentric": 0.0254, "fix": 0.0019, "normal": 0.0002,
}


def _median(values: Sequence[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def _percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (deterministic)."""
    s = sorted(values)
    if not s:
        return 0.0
    if pct <= 0:
        return float(s[0])
    if pct >= 100:
        return float(s[-1])
    import math
    rank = math.ceil(pct / 100.0 * len(s))
    return float(s[min(rank, len(s)) - 1])


def sequence_length_stats(seqs: Sequence[ModelingSequence]) -> Dict[str, float]:
    """Mean / median / p95 token length over a set of sequences."""
    lengths = [token_estimate(s, include_constraints=True) for s in seqs]
    if not lengths:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "count": 0}
    return {
        "mean": sum(lengths) / len(lengths),
        "median": _median(lengths),
        "p95": _percentile(lengths, 95),
        "count": len(lengths),
    }


def constraint_overhead(seqs: Sequence[ModelingSequence]) -> float:
    """Relative token increase from including constraints (>= 0).

    ``(tokens_with_c - tokens_without_c) / tokens_without_c`` aggregated over
    the whole collection. HistCAD reports ~0.327.
    """
    with_c = sum(token_estimate(s, include_constraints=True) for s in seqs)
    without_c = sum(token_estimate(s, include_constraints=False) for s in seqs)
    if without_c == 0:
        return 0.0
    return (with_c - without_c) / without_c


def constraint_distribution(seqs: Sequence[ModelingSequence]) -> Dict[str, float]:
    """Normalised frequency of each constraint type across all sequences."""
    totals = {t: 0 for t in CONSTRAINT_TYPES}
    for s in seqs:
        for feat in s.features:
            hist = constraint_histogram(feat.sketch.constraints)
            for t, n in hist.items():
                totals[t] += n
    grand = sum(totals.values())
    if grand == 0:
        return {t: 0.0 for t in CONSTRAINT_TYPES}
    return {t: totals[t] / grand for t in CONSTRAINT_TYPES}


def total_variation(dist: Dict[str, float],
                    reference: Dict[str, float] = REFERENCE_DISTRIBUTION) -> float:
    """Total-variation distance between two constraint distributions in [0,1]."""
    keys = set(dist) | set(reference)
    return 0.5 * sum(abs(dist.get(k, 0.0) - reference.get(k, 0.0)) for k in keys)


def flattening_ratio(faces: Sequence[Sequence[Sequence]]) -> float:
    """Fraction of primitives removed by flattening a face-loop hierarchy.

    ``1 - |flat_deduped| / |all_primitives_with_multiplicity|``. A hierarchy
    with shared boundaries yields a positive ratio; a hierarchy with no
    duplicates yields 0.
    """
    total = 0
    all_loops: List[Sequence] = []
    for face in faces:
        for loop in face:
            all_loops.append(loop)
            total += len(list(loop))
    if total == 0:
        return 0.0
    flat = symmetric_difference(all_loops)
    return 1.0 - (len(flat) / total)


def hierarchy_free(seqs: Sequence[ModelingSequence]) -> float:
    """Fraction of sequences whose sketches contain no nested (hole) loops.

    A flat HistCAD sequence stores primitives as an unordered set; here we
    approximate 'nesting' by whether any feature declares more than one closed
    loop that could form a containment hierarchy. Provided as a dataset-level
    interpretability proxy. Returns 1.0 for an empty collection.
    """
    from harnesscad.domain.reconstruction.sequences.histcad_replay import reconstruct_loops, hierarchical_loops
    if not seqs:
        return 1.0
    flat_count = 0
    for s in seqs:
        nested = False
        for feat in s.features:
            loops = reconstruct_loops(feat.sketch.primitives)
            loop_dict, _ = hierarchical_loops(loops)
            if any(node.holes for node in loop_dict.values()):
                nested = True
                break
        if not nested:
            flat_count += 1
    return flat_count / len(seqs)
