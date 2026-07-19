"""Dataset difficulty stratification and confusable-pair diagnostics for AFR.

Supplementary deterministic tooling for feature-recognition evaluation.

The core feature-recognition machinery -- the hierarchical manufacturing-feature
taxonomy, the rule-based / model-driven feature detector, the four evaluation
metrics (feature name accuracy, feature quantity accuracy, hallucination rate,
mean absolute error) and the per-feature attribute extraction -- is implemented
elsewhere in the codebase (the ``mfgfeat_*`` modules for the primary
pass, plus ``bench/engdesign_dfm_scoring.py`` for DFM feature-recognition
metrics). This module deliberately does NOT re-implement any of those.

Instead it captures two smaller, self-contained ideas that the
detector / taxonomy / metric layer does not express:

1.  **Difficulty stratification**. A 100-part labelled dataset is partitioned
    into ``easy`` / ``medium`` / ``hard`` buckets using
    explicit, human-auditable rules based on feature *count* and the *presence*
    of particular complex feature families. Easy parts have fewer than six
    features and *exclude* gussets, ribs, necks, threaded features and sheet
    metal features; medium parts add a limited presence of complex features
    such as sheet-metal bends and threaded components; hard parts contain
    numerous features including casting (draft) and freeform (depression /
    protrusion) features. :func:`classify_difficulty` re-implements those rules
    deterministically so any labelled part can be assigned a difficulty tier
    for stratified reporting.

2.  **Confusable-pair diagnostics**. Across every difficulty
    level, recognisers systematically swap *visually
    similar* feature pairs -- ``chamfer`` vs ``fillet`` and ``pipe/tube`` vs
    ``boss``. :func:`confusion_report` takes a ground-truth count vector and a
    prediction count vector and quantifies, per known-confusable pair, how much
    of the error is explained by leakage between the two members (one member
    under-counted while the sibling is over-counted). This is a diagnostic that
    complements -- but is orthogonal to -- the aggregate metrics: MAE
    tells you *how wrong*, this tells you *whether the wrongness is a mix-up*.

Everything is stdlib-only, deterministic (no wall clock, no randomness) and
operates on plain ``dict``/``str`` count vectors, so it has no dependency on any
geometry kernel or on the primary ``mfgfeat_*`` modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Tuple


# --------------------------------------------------------------------------- #
# Feature vocabulary (canonical lowercase keys)
# --------------------------------------------------------------------------- #
# The taxonomy groups manufacturing features into five
# primary categories. For the two diagnostics here we only need the flat set of
# leaf feature keys plus a few semantic groupings used by the difficulty rules.

#: Features whose presence disqualifies a part from the "easy" tier
#: (easy excludes gussets, ribs, necks, threaded features
#: and sheet metal features).
EASY_EXCLUDED_FEATURES: frozenset = frozenset(
    {
        "gusset",
        "rib",
        "neck",
        "thread",
        "gear_teeth",
        "sheet_metal",
        "sheet_metal_bend",
    }
)

#: Features that push a part to the "hard" tier -- casting (draft) and freeform
#: (depression / protrusion) features. The difficult level is characterised by
#: highly complex designs, including intricate geometries such as casting and
#: freeform features.
HARD_INDICATOR_FEATURES: frozenset = frozenset(
    {
        "draft",
        "depression",
        "protrusion",
    }
)

#: Pairs of features that recognisers frequently confuse because of
#: their subtle visual similarity (chamfer/fillet, pipe-tube/boss).
#: Order within a pair is irrelevant.
CONFUSABLE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("chamfer", "fillet"),
    ("pipe_tube", "boss"),
)

#: Default count threshold separating "few" from "numerous" features. Easy
#: parts have fewer than six features; a part at or
#: above ``HARD_COUNT_THRESHOLD`` is treated as "numerous".
EASY_MAX_FEATURE_COUNT: int = 6
HARD_COUNT_THRESHOLD: int = 15


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #
def _normalise_key(name: str) -> str:
    """Canonicalise a feature name to a lowercase underscore key.

    Accepts human labels such as ``"Pipe/Tube"``, ``"Sheet Metal"`` or
    ``"gear teeth"`` and maps them to ``"pipe_tube"``, ``"sheet_metal"`` and
    ``"gear_teeth"`` respectively.
    """
    key = name.strip().lower()
    for ch in (" ", "/", "-", "&"):
        key = key.replace(ch, "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key.strip("_")


def normalise_counts(counts: Mapping[str, int]) -> Dict[str, int]:
    """Return a canonicalised copy of *counts*, summing colliding keys.

    Non-positive quantities are dropped so that presence tests are unambiguous.
    Raises ``ValueError`` on a negative quantity.
    """
    out: Dict[str, int] = {}
    for raw, qty in counts.items():
        if qty < 0:
            raise ValueError(f"negative feature quantity for {raw!r}: {qty}")
        if qty == 0:
            continue
        key = _normalise_key(raw)
        out[key] = out.get(key, 0) + int(qty)
    return out


def total_feature_quantity(counts: Mapping[str, int]) -> int:
    """Total number of feature *instances* (sum of quantities)."""
    return sum(int(q) for q in counts.values() if q > 0)


def distinct_feature_count(counts: Mapping[str, int]) -> int:
    """Number of *distinct* feature types present (quantity > 0)."""
    return len(normalise_counts(counts))


# --------------------------------------------------------------------------- #
# 1. Difficulty stratification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DifficultyResult:
    """Outcome of :func:`classify_difficulty`.

    Attributes
    ----------
    level:
        ``"easy"``, ``"medium"`` or ``"hard"``.
    total_quantity:
        Sum of feature quantities on the part.
    excluded_present:
        Sorted list of easy-excluded features present (rib/gusset/... ).
    hard_present:
        Sorted list of hard-indicator features present (draft/freeform).
    reasons:
        Human-readable justification strings for the assigned level.
    """

    level: str
    total_quantity: int
    excluded_present: Tuple[str, ...]
    hard_present: Tuple[str, ...]
    reasons: Tuple[str, ...]


def classify_difficulty(
    counts: Mapping[str, int],
    *,
    easy_max_count: int = EASY_MAX_FEATURE_COUNT,
    hard_count_threshold: int = HARD_COUNT_THRESHOLD,
) -> DifficultyResult:
    """Assign an easy/medium/hard tier to a labelled CAD part.

    Deterministic stratification rules:

    * **hard** if the part contains any casting/freeform (hard-indicator)
      feature, or its total feature quantity reaches ``hard_count_threshold``
      ("numerous features");
    * else **medium** if it contains any easy-excluded complex feature
      (rib / gusset / neck / thread / gear teeth / sheet metal), or its total
      feature quantity reaches ``easy_max_count`` (no longer "fewer than six");
    * else **easy**.

    The thresholds are exposed as parameters so callers can reproduce
    alternative stratifications; the defaults are the standard thresholds.
    """
    if easy_max_count < 1:
        raise ValueError("easy_max_count must be >= 1")
    if hard_count_threshold < easy_max_count:
        raise ValueError("hard_count_threshold must be >= easy_max_count")

    norm = normalise_counts(counts)
    total = total_feature_quantity(norm)
    excluded = tuple(sorted(k for k in norm if k in EASY_EXCLUDED_FEATURES))
    hard = tuple(sorted(k for k in norm if k in HARD_INDICATOR_FEATURES))

    reasons: List[str] = []

    if hard:
        reasons.append("contains casting/freeform features: " + ", ".join(hard))
        return DifficultyResult("hard", total, excluded, hard, tuple(reasons))
    if total >= hard_count_threshold:
        reasons.append(
            f"numerous features (total quantity {total} >= {hard_count_threshold})"
        )
        return DifficultyResult("hard", total, excluded, hard, tuple(reasons))

    if excluded:
        reasons.append(
            "contains complex features excluded from 'easy': " + ", ".join(excluded)
        )
        return DifficultyResult("medium", total, excluded, hard, tuple(reasons))
    if total >= easy_max_count:
        reasons.append(
            f"feature quantity {total} >= {easy_max_count} (not 'fewer than six')"
        )
        return DifficultyResult("medium", total, excluded, hard, tuple(reasons))

    reasons.append(
        f"fewer than {easy_max_count} features and no complex feature families"
    )
    return DifficultyResult("easy", total, excluded, hard, tuple(reasons))


def stratify_dataset(
    parts: Mapping[str, Mapping[str, int]],
    **kwargs,
) -> Dict[str, List[str]]:
    """Group named parts into ``easy``/``medium``/``hard`` buckets.

    *parts* maps a part id to its ground-truth feature count vector. Returns a
    dict with keys ``"easy"``, ``"medium"``, ``"hard"``; each value is the
    sorted list of part ids in that tier. Extra keyword arguments are forwarded
    to :func:`classify_difficulty`.
    """
    buckets: Dict[str, List[str]] = {"easy": [], "medium": [], "hard": []}
    for part_id, counts in parts.items():
        level = classify_difficulty(counts, **kwargs).level
        buckets[level].append(part_id)
    for level in buckets:
        buckets[level].sort()
    return buckets


# --------------------------------------------------------------------------- #
# 2. Confusable-pair diagnostics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairConfusion:
    """Confusion diagnostic for a single confusable feature pair.

    Attributes
    ----------
    pair:
        The two canonical feature keys, in a stable sorted order.
    over:
        For each member, ``predicted - ground_truth`` when positive (i.e. how
        much the member was over-predicted), else 0.
    under:
        For each member, ``ground_truth - predicted`` when positive (i.e. how
        much the member was under-predicted), else 0.
    swap_magnitude:
        Amount of the error explainable as a *swap* between the two members:
        ``min(over[a], under[b]) + min(over[b], under[a])``. A large value means
        the model is trading counts between the two look-alike features.
    """

    pair: Tuple[str, str]
    over: Dict[str, int]
    under: Dict[str, int]
    swap_magnitude: int


def confusion_report(
    ground_truth: Mapping[str, int],
    predicted: Mapping[str, int],
    *,
    pairs: Iterable[Tuple[str, str]] = CONFUSABLE_PAIRS,
) -> Dict[Tuple[str, str], PairConfusion]:
    """Quantify swap-confusion for each known confusable feature pair.

    For every pair ``(a, b)`` the over/under-prediction of each member is
    computed from the two count vectors, and the ``swap_magnitude`` measures how
    much of the discrepancy is consistent with the model confusing ``a`` for
    ``b`` (or vice versa). Result is keyed by the sorted pair tuple.
    """
    gt = normalise_counts(ground_truth)
    pr = normalise_counts(predicted)

    report: Dict[Tuple[str, str], PairConfusion] = {}
    for raw_a, raw_b in pairs:
        a = _normalise_key(raw_a)
        b = _normalise_key(raw_b)
        key = tuple(sorted((a, b)))

        over: Dict[str, int] = {}
        under: Dict[str, int] = {}
        for member in (a, b):
            diff = pr.get(member, 0) - gt.get(member, 0)
            over[member] = diff if diff > 0 else 0
            under[member] = -diff if diff < 0 else 0

        swap = min(over[a], under[b]) + min(over[b], under[a])
        report[key] = PairConfusion(
            pair=(key[0], key[1]),
            over=over,
            under=under,
            swap_magnitude=swap,
        )
    return report


def total_swap_confusion(
    ground_truth: Mapping[str, int],
    predicted: Mapping[str, int],
    *,
    pairs: Iterable[Tuple[str, str]] = CONFUSABLE_PAIRS,
) -> int:
    """Sum of :attr:`PairConfusion.swap_magnitude` over all pairs."""
    return sum(
        pc.swap_magnitude
        for pc in confusion_report(
            ground_truth, predicted, pairs=pairs
        ).values()
    )
