"""SynthBal: synthetic dataset balancing across CAD-program complexity.

Yu, Alam, Hart & Ahmed, *CAD Program Generation using GenCAD-3D: Multimodal
Latent Space Alignment and Synthetic Dataset Balancing* (J. Mech. Des. 2026),
Section 6 and Algorithm 1.

The DeepCAD corpus is heavily skewed by **complexity**, defined as the CAD
program's sequence length ell (number of commands): sequence length <= 6 makes up
42 % of the data and exactly 6 makes up 26 %, while long programs (up to length
59) are almost absent (paper Fig. 4). A model trained on that skew reconstructs
simple prisms/cylinders well but fails on intricate parts. Prior augmentation
(DeepCAD random-replacement, ContrastCAD RRE -- see ``datagen/contrastcad_rre``)
increases *diversity* but does not correct the *proportion* imbalance.

SynthBal rebalances by target: it produces a synthetic dataset ``S`` of size
``N_S`` with an **equal number of programs per sequence length**. For each length
ell it (a) copies up to a ratio ``r`` of real data (``min(r*n_Sl, n_Rl)``), then
(b) fills the remainder with *validated* augmentations of real programs of that
length. Unlike train-time augmentation, augmentations are generated offline so
each can be validity-checked before admission (paper: this is why SynthBal can
enforce validity where DeepCAD/ContrastCAD cannot).

This module implements the deterministic pieces:

* ``perturb_noise`` -- the *Noise* building block: perturb a ratio ``p`` of
  commands, adding uniform noise in ``[-m*256, m*256]`` to continuous parameters
  only (discrete / near-discrete values such as extrude orientation are skipped),
  clipping to ``[1, 255]``.
* ``replace_sketch`` -- the *Replace-Sketch* building block: swap sketch loops of
  one program for those of a donor program (parses pairs via
  ``datagen.contrastcad_rre.split_pairs``).
* ``synthbal_augment`` -- the weighted augmentation choice (40 % large pure-noise
  ``m=0.07, p=0.6``; else 60 % small-noise ``m=0.02, p=0.8`` then Replace-Sketch).
* ``balance_dataset`` / ``synthbal_dataset`` -- Algorithm 1 itself.
* ``reduction_balance`` -- the SYNAuG "reduction-balanced" pass: purely-real
  balancing by *removing* from over-represented lengths.
* imbalance metrics: ``length_histogram``, ``class_shares``, ``imbalance_ratio``,
  ``balanced_target``, ``imbalance_report``.

The geometric validity predicate (OpenCascade compile + self-intersection check)
is external; ``structural_valid`` is a deterministic stdlib stand-in and the
predicate is caller-pluggable. Determinism: all randomness flows through a
supplied ``random.Random`` or integer seed; stdlib only; no wall clock.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from datagen.contrastcad_rre import (
    ARC,
    CIRCLE,
    EOS,
    EXTRUDE,
    LINE,
    SOL,
    split_pairs,
)

Command = Dict[str, object]
Program = List[Command]

# Continuous parameters are perturbable by Noise; discrete / near-discrete
# parameters (flags, extrude type, boolean op, orientation angles which the paper
# notes are typically 0 / +-90 deg) are left untouched.
DEFAULT_DISCRETE_KEYS = frozenset({
    "type", "c", "w", "boolean", "op",
    "orientation", "theta_o", "phi", "gamma", "sign",
})

QUANT_LO = 1
QUANT_HI = 255
SKETCH_PRIMITIVES = frozenset({LINE, ARC, CIRCLE})


def _rng(seed) -> random.Random:
    if isinstance(seed, random.Random):
        return seed
    return random.Random(seed)


def _length_of(program: Sequence[Command]) -> int:
    """Sequence length ell = number of commands (paper Sec. 4.1)."""
    return len(program)


# =====================================================================
# Augmentation building block 1: Noise
# =====================================================================

def perturb_noise(program: Sequence[Command], seed,
                  magnitude: float = 0.02, prob: float = 0.8,
                  discrete_keys=DEFAULT_DISCRETE_KEYS) -> Program:
    """Perturb continuous parameters of a ratio ``prob`` of commands (paper Sec. 6).

    Each command is independently selected with probability ``prob``. For a
    selected command, every numeric parameter whose key is *not* in
    ``discrete_keys`` gets additive noise drawn uniformly from
    ``[-magnitude*256, magnitude*256]``, rounded to an integer and clipped to
    ``[1, 255]``. Discrete / near-discrete values (flags, extrude type/orientation)
    are preserved. Deterministic given ``seed``.
    """
    if not 0.0 <= prob <= 1.0:
        raise ValueError("prob must be in [0, 1]")
    if magnitude < 0.0:
        raise ValueError("magnitude must be non-negative")
    rng = _rng(seed)
    span = magnitude * 256.0
    out: Program = []
    for cmd in program:
        new = dict(cmd)
        if rng.random() < prob:
            for key, val in list(new.items()):
                if key in discrete_keys:
                    continue
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    continue
                perturbed = val + rng.uniform(-span, span)
                new[key] = int(max(QUANT_LO, min(QUANT_HI, round(perturbed))))
        out.append(new)
    return out


# =====================================================================
# Augmentation building block 2: Replace-Sketch
# =====================================================================

def _sketch_portion(pair: Sequence[Command]) -> List[int]:
    """Indices of the sketch commands (everything but a trailing extrude) in a pair."""
    idx = list(range(len(pair)))
    if pair and pair[-1].get("type") == EXTRUDE:
        return idx[:-1]
    return idx


def replace_sketch(program: Sequence[Command], donor: Sequence[Command], seed,
                   replace_prob: float = 0.5) -> Program:
    """Replace sketch loops of ``program`` with those of ``donor`` (paper Sec. 6).

    Both programs are split into sketch-and-extrude pairs. Each extrusion-terminated
    pair of ``program`` is, with probability ``replace_prob``, given the sketch
    commands of a randomly chosen extrusion pair of ``donor`` while keeping its own
    extrude command (so the sketch changes but the solid operation is preserved).
    If the donor has no extrusion pair the program is returned unchanged.
    Deterministic given ``seed``.
    """
    if not 0.0 <= replace_prob <= 1.0:
        raise ValueError("replace_prob must be in [0, 1]")
    rng = _rng(seed)
    mine = split_pairs(program)
    theirs = split_pairs(donor)
    donor_pairs = [p for p in theirs if p and p[-1].get("type") == EXTRUDE]
    if not donor_pairs:
        return [dict(c) for group in mine for c in group]
    for i, pair in enumerate(mine):
        if not (pair and pair[-1].get("type") == EXTRUDE):
            continue
        if rng.random() < replace_prob:
            src = donor_pairs[rng.randrange(len(donor_pairs))]
            src_sketch = [dict(src[j]) for j in _sketch_portion(src)]
            extrude = dict(pair[-1])
            mine[i] = src_sketch + [extrude]
    return [dict(c) for group in mine for c in group]


# =====================================================================
# Weighted augmentation choice (paper Sec. 6)
# =====================================================================

def synthbal_augment(program: Sequence[Command], donor: Sequence[Command],
                     seed) -> Program:
    """SynthBal's default augmentation: weighted large-noise vs small-noise+replace.

    Paper Sec. 6: with 40 % probability apply large pure-noise (``m=0.07, p=0.6``);
    otherwise (60 %) apply small noise (``m=0.02, p=0.8``) then a Replace-Sketch
    against ``donor``. Deterministic given ``seed``.
    """
    rng = _rng(seed)
    if rng.random() < 0.40:
        return perturb_noise(program, rng.randint(0, 2 ** 31 - 1),
                             magnitude=0.07, prob=0.6)
    noised = perturb_noise(program, rng.randint(0, 2 ** 31 - 1),
                           magnitude=0.02, prob=0.8)
    return replace_sketch(noised, donor, rng.randint(0, 2 ** 31 - 1))


# =====================================================================
# Validity predicate (structural stand-in; geometric check is external)
# =====================================================================

def structural_valid(program: Sequence[Command]) -> bool:
    """Deterministic structural validity: at least one sketch primitive and one
    extrude, with every continuous parameter inside ``[0, 255]``.

    The paper's real predicate also compiles the program with OpenCascade and
    rejects self-intersections -- that geometric check is external; supply your own
    predicate to ``balance_dataset`` to use it.
    """
    if not program:
        return False
    has_sketch = any(c.get("type") in SKETCH_PRIMITIVES for c in program)
    has_extrude = any(c.get("type") == EXTRUDE for c in program)
    if not (has_sketch and has_extrude):
        return False
    for c in program:
        for key, val in c.items():
            if key in DEFAULT_DISCRETE_KEYS:
                continue
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                continue
            if not 0 <= val <= 255:
                return False
    return True


# =====================================================================
# Imbalance metrics
# =====================================================================

def length_histogram(dataset: Sequence[Sequence[Command]],
                     length_of: Callable = _length_of) -> Dict[int, int]:
    """Map sequence length -> number of programs of that length."""
    hist: Dict[int, int] = defaultdict(int)
    for prog in dataset:
        hist[length_of(prog)] += 1
    return dict(sorted(hist.items()))


def class_shares(histogram: Dict[int, int]) -> Dict[int, float]:
    """Per-length proportion of the dataset (sums to 1.0 over present lengths)."""
    total = sum(histogram.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in sorted(histogram.items())}


def imbalance_ratio(histogram: Dict[int, int]) -> float:
    """Max/min bucket-count ratio over present lengths (1.0 = perfectly balanced).

    ``inf`` when some present length has zero count (cannot happen for a histogram
    built from data, but guarded for externally supplied histograms).
    """
    counts = [v for v in histogram.values()]
    if not counts:
        return 0.0
    lo = min(counts)
    if lo == 0:
        return float("inf")
    return max(counts) / lo


def balanced_target(lengths: Sequence[int]) -> Dict[int, float]:
    """Uniform target distribution over the given lengths (``n_Sl = N_S/|L|``)."""
    uniq = sorted(set(lengths))
    if not uniq:
        return {}
    w = 1.0 / len(uniq)
    return {ell: w for ell in uniq}


@dataclass
class BalanceReport:
    """Per-length accounting of a SynthBal run."""

    target_size: int
    per_length_target: int
    lengths: List[int] = field(default_factory=list)
    real_counts: Dict[int, int] = field(default_factory=dict)
    synthetic_counts: Dict[int, int] = field(default_factory=dict)
    produced_counts: Dict[int, int] = field(default_factory=dict)
    rejected: int = 0

    @property
    def total_produced(self) -> int:
        return sum(self.produced_counts.values())

    @property
    def total_real(self) -> int:
        return sum(self.real_counts.values())

    @property
    def total_synthetic(self) -> int:
        return sum(self.synthetic_counts.values())

    def synthetic_fraction(self) -> float:
        n = self.total_produced
        return (self.total_synthetic / n) if n else 0.0

    def to_dict(self) -> dict:
        return {
            "target_size": self.target_size,
            "per_length_target": self.per_length_target,
            "lengths": list(self.lengths),
            "real_counts": dict(self.real_counts),
            "synthetic_counts": dict(self.synthetic_counts),
            "produced_counts": dict(self.produced_counts),
            "rejected": self.rejected,
            "total_produced": self.total_produced,
            "total_real": self.total_real,
            "total_synthetic": self.total_synthetic,
            "synthetic_fraction": self.synthetic_fraction(),
        }


def imbalance_report(dataset: Sequence[Sequence[Command]],
                     length_of: Callable = _length_of) -> dict:
    """Summarise a dataset's complexity imbalance (histogram, shares, ratio)."""
    hist = length_histogram(dataset, length_of)
    return {
        "histogram": hist,
        "shares": class_shares(hist),
        "imbalance_ratio": imbalance_ratio(hist),
        "n_lengths": len(hist),
        "n_items": sum(hist.values()),
    }


# =====================================================================
# Algorithm 1: SynthBal dataset generation
# =====================================================================

def balance_dataset(dataset: Sequence[Program],
                    target_size: int,
                    real_ratio: float,
                    seed,
                    augment: Optional[Callable] = None,
                    is_valid: Optional[Callable] = None,
                    length_of: Callable = _length_of,
                    max_tries_per_slot: int = 32):
    """SynthBal (paper Algorithm 1): build a length-balanced dataset of size N_S.

    For each present sequence length ``ell`` the per-length target is
    ``n_Sl = target_size // |L|``. First ``min(real_ratio*n_Sl, n_Rl)`` real
    programs of that length are copied (sampled without replacement); the rest are
    synthesised by repeatedly augmenting a random real program of that length and
    admitting the result only if ``is_valid`` accepts it.

    * ``augment(program, donor, rng)`` defaults to :func:`synthbal_augment`; the
      donor is a random program of the same length.
    * ``is_valid`` defaults to :func:`structural_valid`.
    * ``max_tries_per_slot`` bounds rejected augmentations per needed sample so an
      always-invalid augmenter terminates.

    Returns ``(balanced, report)`` where ``balanced`` is the list of programs and
    ``report`` is a :class:`BalanceReport`. Deterministic given ``seed``.
    """
    if target_size < 0:
        raise ValueError("target_size must be non-negative")
    if not 0.0 <= real_ratio <= 1.0:
        raise ValueError("real_ratio must be in [0, 1]")
    augment = augment or synthbal_augment
    is_valid = is_valid or structural_valid
    rng = _rng(seed)

    by_len: Dict[int, List[Program]] = defaultdict(list)
    for prog in dataset:
        by_len[length_of(prog)].append(prog)
    lengths = sorted(by_len)

    report = BalanceReport(target_size=target_size,
                           per_length_target=(target_size // len(lengths)) if lengths else 0,
                           lengths=lengths)
    if not lengths:
        return [], report

    n_per = target_size // len(lengths)
    balanced: List[Program] = []

    for ell in lengths:
        pool = by_len[ell]
        n_real = min(int(real_ratio * n_per), len(pool))
        reals = rng.sample(pool, n_real) if n_real else []
        slot: List[Program] = [[dict(c) for c in p] for p in reals]
        real_admitted = len(slot)
        synth = 0
        while len(slot) < n_per:
            tries = 0
            admitted = False
            while tries < max_tries_per_slot:
                base = pool[rng.randrange(len(pool))]
                donor = pool[rng.randrange(len(pool))]
                cand = augment(base, donor, rng.randint(0, 2 ** 31 - 1))
                tries += 1
                if is_valid(cand):
                    slot.append(cand)
                    synth += 1
                    admitted = True
                    break
                report.rejected += 1
            if not admitted:
                # Could not fill this slot within the try budget; stop for this length.
                break
        report.real_counts[ell] = real_admitted
        report.synthetic_counts[ell] = synth
        report.produced_counts[ell] = len(slot)
        balanced.extend(slot)

    return balanced, report


def synthbal_dataset(dataset: Sequence[Program], target_size: int, seed,
                     real_ratio: float = 0.2, is_valid: Optional[Callable] = None):
    """Convenience wrapper for :func:`balance_dataset` with the paper's default
    ``r = 0.2`` and the weighted :func:`synthbal_augment` scheme (SynthBal set)."""
    return balance_dataset(dataset, target_size, real_ratio, seed,
                           augment=synthbal_augment, is_valid=is_valid)


# =====================================================================
# SYNAuG reduction-balanced pass (purely real, remove over-represented)
# =====================================================================

def reduction_balance(dataset: Sequence[Program], seed,
                      length_of: Callable = _length_of,
                      per_length: Optional[int] = None) -> List[Program]:
    """Purely-real balancing by removing from over-represented lengths (paper Sec. 6).

    Each present length is down-sampled (without replacement) to ``per_length``
    programs; ``per_length`` defaults to the smallest present bucket count so every
    length ends up equally represented using only real data. Used by GenCAD-3D for
    the reduction-balanced fine-tuning set. Deterministic given ``seed``.
    """
    rng = _rng(seed)
    by_len: Dict[int, List[Program]] = defaultdict(list)
    for prog in dataset:
        by_len[length_of(prog)].append(prog)
    if not by_len:
        return []
    target = per_length if per_length is not None else min(len(v) for v in by_len.values())
    out: List[Program] = []
    for ell in sorted(by_len):
        pool = by_len[ell]
        take = min(target, len(pool))
        out.extend(rng.sample(pool, take))
    return out
