"""Sequence-length-normalized evaluation for CAD reconstruction metrics.

Yu, Alam, Hart & Ahmed, *CAD Program Generation using GenCAD-3D* (J. Mech. Des.
2026), Section 7 ("Sequence-Length Normalization", Eq. 6) and the "Comparing
Accuracy" relative-improvement convention.

Most CAD studies report a metric averaged over the whole test set. When the set
is complexity-imbalanced (DeepCAD is dominated by sequence lengths 5-15), that
average mostly reflects the over-represented short programs and *hides* how the
model does on long, under-represented ones (paper Table 1). GenCAD-3D fixes this
by averaging the metric across sequence-length buckets instead of across items:

    m_SL(X) = (1/|L|) * sum over ell in L of  m(X_ell)                    (Eq. 6)

where ``X_ell`` are the items of sequence length ``ell`` and ``L`` is the set of
present lengths. Every length contributes equally, so a length appearing once
counts as much as one appearing ten-thousand times -- surfacing performance on
the hard cases.

The module also implements the paper's **relative-error-improvement** convention
for comparing two accuracies: ``p1`` improves on ``p2`` by ``(p1 - p2)/(1 - p2)``,
so 0.995 vs 0.990 and 0.95 vs 0.90 both read as a 50 % improvement -- better
reflecting order-of-magnitude gains near the ceiling.

This complements ``bench/gencad_retrieval`` (batched R_B accuracy) and
``bench/contrastcad_recon_accuracy`` (per-item command/parameter accuracy): those
compute the *raw* metrics, this reweights any per-item metric by sequence length.
Pure stdlib, deterministic (bucket key order is sorted; no randomness, no clock).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class NormalizedMetric:
    """Result of sequence-length normalization for one metric."""

    unnormalized: float               # m(X): mean over all items
    normalized: float                 # m_SL(X): mean over length buckets (Eq. 6)
    per_length: Dict[int, float] = field(default_factory=dict)  # ell -> mean(X_ell)
    counts: Dict[int, int] = field(default_factory=dict)        # ell -> |X_ell|
    higher_is_better: bool = True

    @property
    def bias(self) -> float:
        """``unnormalized - normalized``: how much the raw average is inflated
        (for higher-is-better metrics) by over-represented short sequences."""
        return self.unnormalized - self.normalized

    def worst_length(self):
        """The sequence length with the least-favourable per-length value."""
        if not self.per_length:
            return None
        if self.higher_is_better:
            return min(self.per_length, key=lambda k: self.per_length[k])
        return max(self.per_length, key=lambda k: self.per_length[k])

    def to_dict(self) -> dict:
        return {
            "unnormalized": self.unnormalized,
            "normalized": self.normalized,
            "bias": self.bias,
            "per_length": dict(sorted(self.per_length.items())),
            "counts": dict(sorted(self.counts.items())),
            "higher_is_better": self.higher_is_better,
            "worst_length": self.worst_length(),
        }


def _validate(values: Sequence[float], seq_lengths: Sequence[int]) -> None:
    if len(values) != len(seq_lengths):
        raise ValueError("values and seq_lengths must have equal length")
    if not values:
        raise ValueError("at least one item is required")


def per_length_means(values: Sequence[float],
                     seq_lengths: Sequence[int]) -> Dict[int, float]:
    """Mean of ``values`` grouped by sequence length (``m(X_ell)`` for each ell)."""
    _validate(values, seq_lengths)
    sums: Dict[int, float] = defaultdict(float)
    counts: Dict[int, int] = defaultdict(int)
    for v, ell in zip(values, seq_lengths):
        sums[ell] += float(v)
        counts[ell] += 1
    return {ell: sums[ell] / counts[ell] for ell in sorted(sums)}


def sequence_length_normalized(values: Sequence[float],
                               seq_lengths: Sequence[int],
                               higher_is_better: bool = True) -> NormalizedMetric:
    """Compute ``m(X)`` and ``m_SL(X)`` (Eq. 6) for a per-item metric.

    ``values[i]`` is the metric for item ``i`` (e.g. its command accuracy, or a
    0/1 invalid flag); ``seq_lengths[i]`` is that item's CAD-program sequence
    length. ``m_SL`` averages the per-length means, giving each present length
    equal weight. Returns a :class:`NormalizedMetric`.
    """
    _validate(values, seq_lengths)
    unnorm = sum(float(v) for v in values) / len(values)
    counts: Dict[int, int] = defaultdict(int)
    for ell in seq_lengths:
        counts[ell] += 1
    per_len = per_length_means(values, seq_lengths)
    norm = sum(per_len.values()) / len(per_len)
    return NormalizedMetric(
        unnormalized=unnorm,
        normalized=norm,
        per_length=per_len,
        counts=dict(sorted(counts.items())),
        higher_is_better=higher_is_better,
    )


def relative_error_improvement(p1: float, p2: float) -> float:
    """Relative improvement of accuracy ``p1`` over baseline ``p2`` (paper Sec. 7).

    ``(p1 - p2) / (1 - p2)``: the fraction of the *remaining* error that ``p1``
    closes relative to ``p2``. Positive means ``p1`` is better. ``p1``/``p2`` are
    accuracies in ``[0, 1]``; ``p2`` must not be exactly 1.0.
    """
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError("accuracies must be in [0, 1]")
    if p2 == 1.0:
        raise ValueError("baseline accuracy p2 must be < 1.0")
    return (p1 - p2) / (1.0 - p2)


def relative_reduction(e1: float, e2: float) -> float:
    """Relative reduction of an error-like quantity ``e1`` vs baseline ``e2``.

    ``(e2 - e1) / e2`` -- the fraction by which ``e1`` reduces the baseline error
    (positive = improvement). Used for lower-is-better metrics such as invalid
    ratio, chamfer distance, or median error. ``e2`` must be > 0.
    """
    if e2 <= 0.0:
        raise ValueError("baseline e2 must be positive")
    return (e2 - e1) / e2


def compare_normalized(candidate: NormalizedMetric,
                       baseline: NormalizedMetric) -> dict:
    """Relative improvement of ``candidate`` over ``baseline`` on both m and m_SL.

    Chooses the relative-error or relative-reduction convention automatically from
    ``higher_is_better``. For higher-is-better metrics the values are treated as
    accuracies in ``[0, 1]``; for lower-is-better they are treated as errors.
    """
    if candidate.higher_is_better != baseline.higher_is_better:
        raise ValueError("candidate and baseline must share metric direction")
    if candidate.higher_is_better:
        return {
            "unnormalized": relative_error_improvement(candidate.unnormalized,
                                                       baseline.unnormalized),
            "normalized": relative_error_improvement(candidate.normalized,
                                                     baseline.normalized),
        }
    return {
        "unnormalized": relative_reduction(candidate.unnormalized,
                                           baseline.unnormalized),
        "normalized": relative_reduction(candidate.normalized,
                                         baseline.normalized),
    }
