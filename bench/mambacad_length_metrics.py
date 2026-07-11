"""Long-sequence length statistics for Mamba-CAD (Table 1 / Table 2).

Mamba-CAD's whole thesis is that a State Space Model can model *longer*
parametric CAD sequences than Transformer baselines, so the paper reports a
family of **length-centric** statistics that are all deterministic:

* **Average Length (AL)** of valid parametric CAD sequences (Table 1 "AL",
  Table 2 "AL"): the mean effective length.
* **Length distribution** over the buckets ``[1-10] [11-25] [26-40] [41-60]
  [60-128]`` (Table 1): the fraction of sequences whose length falls in each
  bucket.
* **L>=60 ratio** (Table 2 caption): "the ratio of valid reconstructed CAD
  sequences (Length>=60) in total CAD sequences (Length>=60) within test set"
  -- i.e. among ground-truth sequences at least ``threshold`` long, the fraction
  whose reconstruction is valid *and* still at least ``threshold`` long. This is
  the metric that most directly measures long-sequence modelling ability.

These are distinct from the already-present generation metrics
(``bench/diffusioncad_generation_metrics.py`` already provides Unique / Novel /
Invalid ratios, and ``bench/contrastcad_recon_accuracy.py`` provides the Ac/Ap
command/parameter accuracies) -- none of those look at sequence *length*.

A parametric CAD sequence is padded to a fixed ``N`` (128 in the paper) with an
``<EOS>`` marker; the *effective* length is the number of real commands before
the first ``<EOS>``. :func:`effective_length` extracts that; the other functions
consume plain integer lengths so they work regardless of token encoding.

Deterministic, stdlib-only.
"""

from __future__ import annotations

# Table 1 length buckets (inclusive integer ranges). ``None`` upper bound means
# "no upper limit".
DEEPCAD_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("1-10", 1, 10),
    ("11-25", 11, 25),
    ("26-40", 26, 40),
    ("41-60", 41, 60),
    ("60-128", 61, 128),
)


def effective_length(sequence, eos_token) -> int:
    """Number of real commands before the first ``eos_token``.

    Trailing ``<EOS>`` padding (paper "CAD sequence representation") is not
    counted. If no ``eos_token`` is present the full sequence length is returned.
    """
    for i, tok in enumerate(sequence):
        if tok == eos_token:
            return i
    return len(sequence)


def average_length(lengths) -> float:
    """Average Length (AL) of a collection of sequence lengths.

    Returns ``0.0`` for an empty collection.
    """
    lengths = list(lengths)
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)


def length_distribution(lengths, buckets=DEEPCAD_BUCKETS) -> dict[str, float]:
    """Fraction of sequences whose length falls in each bucket (Table 1).

    ``buckets`` is a tuple of ``(label, lo, hi)`` with inclusive integer bounds
    (``hi=None`` means unbounded above). Lengths outside every bucket are
    ignored in the numerator but still counted in the total, so fractions sum to
    <= 1.0 (matching the paper's per-row percentages). Returns fractions in
    ``[0, 1]``; multiply by 100 for the paper's percentages.
    """
    lengths = list(lengths)
    total = len(lengths)
    result: dict[str, float] = {}
    if total == 0:
        for label, _lo, _hi in buckets:
            result[label] = 0.0
        return result
    for label, lo, hi in buckets:
        count = 0
        for n in lengths:
            if n >= lo and (hi is None or n <= hi):
                count += 1
        result[label] = count / total
    return result


def long_sequence_ratio(gt_lengths, recon_valid, recon_lengths,
                        threshold: int = 60) -> float:
    """L>=``threshold`` ratio (Table 2 caption).

    Among ground-truth sequences with ``gt_length >= threshold``, the fraction
    whose reconstruction is *valid* and whose reconstructed length is also
    ``>= threshold``. Returns ``0.0`` when no ground-truth sequence reaches the
    threshold.

    ``recon_valid[i]`` is a boolean (did reconstruction ``i`` produce a
    constructible shape); ``recon_lengths[i]`` is its effective length.
    """
    gt_lengths = list(gt_lengths)
    recon_valid = list(recon_valid)
    recon_lengths = list(recon_lengths)
    if not (len(gt_lengths) == len(recon_valid) == len(recon_lengths)):
        raise ValueError("gt_lengths, recon_valid, recon_lengths must align")
    denom = 0
    numer = 0
    for gl, valid, rl in zip(gt_lengths, recon_valid, recon_lengths):
        if gl >= threshold:
            denom += 1
            if valid and rl >= threshold:
                numer += 1
    if denom == 0:
        return 0.0
    return numer / denom


def length_report(lengths, buckets=DEEPCAD_BUCKETS) -> dict:
    """Deterministic summary: total count, average length, and distribution."""
    lengths = list(lengths)
    return {
        "total": len(lengths),
        "average_length": average_length(lengths),
        "distribution": length_distribution(lengths, buckets),
    }
