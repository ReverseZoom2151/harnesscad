"""Assembly-name task suite: corpus lines, Missing-Part, Document-Name, splits.

Reimplements the deterministic scaffolding of "What's In A Name?" (Meltzer,
Lambourne, Grandi, JCISE 2023) without torch / sklearn / numpy:

* ``corpus_line`` / ``build_corpus`` -- the paper's fine-tuning corpus template
  ``An assembly with name "X" and description "Y", contains the following
  parts: p1, ..., pN.`` (description clause omitted when empty), plus the
  parts-only variant used for FastText.
* ``missing_part_task`` -- leave-one-out: hold out one part of an assembly, the
  model must recover it from the remaining parts. The held-out index is chosen
  by a seeded RNG (the paper draws it randomly per sample); we expose the index
  so the task is reproducible and rankable.
* ``document_name_task`` -- predict the assembly name from the full part set.
* ``stratified_split`` -- a pure-Python replacement for
  ``StratifiedShuffleSplit`` over the paper's five binary document features
  (has-part, >=2 parts, has-feature, parts+features, has-description): documents
  are bucketed by their feature tuple and each bucket is split by the target
  fractions with largest-remainder rounding, so every stratum keeps its
  proportion exactly and no document lands in two splits.
* ``feature_subset_splits`` -- the derived split files
  (``train_val_test_two_or_more_partnames.json`` etc.) as filtered views.
* ``rank_of_target`` / ``retrieval_metrics`` -- accuracy@k and MRR for scoring
  Missing-Part or Document-Name candidate lists.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Mapping, Sequence, Tuple

from harnesscad.domain.library.name_normalizer import (
    CleanDocument,
    clean_corpus,
    document_features,
)

SPLIT_NAMES = ("train", "validation", "test")

FEATURE_KEYS = (
    "partnames",
    "two_or_more_partnames",
    "featurenames",
    "partnames_and_featurenames",
    "descriptions",
)


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


def corpus_line(doc: CleanDocument, *, lower: bool = False) -> str:
    """One fine-tuning corpus line in the paper's template."""
    name_str = f'An assembly with name "{doc.document_name}"'
    desc = doc.document_description.strip()
    desc_str = f' and description "{desc}",' if desc else ""
    parts = list(doc.body_names) + list(doc.feature_names)
    parts_str = f' contains the following parts: {", ".join(parts)}.'
    line = name_str + desc_str + parts_str
    return line.lower() if lower else line


def parts_only_line(
    doc: CleanDocument, *, lower: bool = False, remove_duplicates: bool = False
) -> str:
    """Comma-separated parts line (the FastText corpus variant)."""
    bodies = list(doc.body_names)
    if remove_duplicates:
        seen = set()
        deduped = []
        for b in bodies:
            if b not in seen:
                seen.add(b)
                deduped.append(b)
        bodies = deduped
    line = ", ".join(bodies) + "."
    return line.lower() if lower else line


def build_corpus(
    documents: Sequence[CleanDocument],
    *,
    lower: bool = False,
    parts_only: bool = False,
    remove_duplicates: bool = False,
) -> List[str]:
    """Corpus lines in the given document order (caller controls shuffling)."""
    if parts_only:
        return [
            parts_only_line(d, lower=lower, remove_duplicates=remove_duplicates)
            for d in documents
        ]
    return [corpus_line(d, lower=lower) for d in documents]


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SetTask:
    """A set-in / name-out sample."""

    document_id: str
    inputs: Tuple[str, ...]
    target: str
    kind: str  # "missing_part" or "document_name"


def missing_part_task(
    doc: CleanDocument, *, seed: int = 0
) -> SetTask:
    """Hold out one part; the rest are the input set."""
    parts = list(doc.body_names)
    if len(parts) < 2:
        raise ValueError("missing-part task needs at least 2 parts")
    rng = random.Random(f"{seed}:{doc.document_id}")
    k = rng.randrange(len(parts))
    target = parts[k]
    inputs = tuple(parts[:k] + parts[k + 1 :])
    return SetTask(doc.document_id, inputs, target, "missing_part")


def document_name_task(doc: CleanDocument) -> SetTask:
    """Predict the assembly name from all of its parts."""
    if not doc.body_names:
        raise ValueError("document-name task needs at least 1 part")
    if not doc.document_name:
        raise ValueError("document has no user-provided name")
    return SetTask(
        doc.document_id, tuple(doc.body_names), doc.document_name, "document_name"
    )


def build_tasks(
    documents: Sequence[CleanDocument], *, kind: str = "missing_part", seed: int = 0
) -> List[SetTask]:
    """All feasible tasks of ``kind``, skipping documents that cannot supply one."""
    tasks: List[SetTask] = []
    for doc in documents:
        try:
            if kind == "missing_part":
                tasks.append(missing_part_task(doc, seed=seed))
            elif kind == "document_name":
                tasks.append(document_name_task(doc))
            else:
                raise ValueError(f"unknown task kind: {kind}")
        except ValueError as exc:
            if "unknown task kind" in str(exc):
                raise
            continue
    return tasks


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------


def _largest_remainder(total: int, fractions: Sequence[float]) -> List[int]:
    """Split ``total`` into counts matching ``fractions`` (sums exactly)."""
    raw = [total * f for f in fractions]
    counts = [int(x) for x in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(len(fractions)), key=lambda i: (-(raw[i] - counts[i]), i)
    )
    for i in range(remainder):
        counts[order[i % len(order)]] += 1
    return counts


def stratified_split(
    documents: Mapping[str, CleanDocument],
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 9876,
) -> Dict[str, List[str]]:
    """Stratify document ids by the five binary features, then split.

    Each stratum (a distinct feature tuple) is shuffled with the seeded RNG and
    apportioned train/validation/test by largest-remainder rounding.
    """
    if not 0.0 <= validation_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("fractions must lie in [0, 1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation + test fractions must be < 1")

    strata: Dict[Tuple[int, ...], List[str]] = {}
    for doc_id in sorted(documents):
        key = document_features(documents[doc_id])
        strata.setdefault(key, []).append(doc_id)

    train_fraction = 1.0 - validation_fraction - test_fraction
    fractions = (train_fraction, validation_fraction, test_fraction)
    out: Dict[str, List[str]] = {name: [] for name in SPLIT_NAMES}
    for key in sorted(strata):
        ids = strata[key]
        rng = random.Random(f"{seed}:{key}")
        shuffled = rng.sample(ids, k=len(ids))
        counts = _largest_remainder(len(shuffled), fractions)
        cursor = 0
        for name, count in zip(SPLIT_NAMES, counts):
            out[name].extend(shuffled[cursor : cursor + count])
            cursor += count
    for name in SPLIT_NAMES:
        out[name].sort()
    return out


def feature_subset_splits(
    documents: Mapping[str, CleanDocument], split: Mapping[str, Sequence[str]]
) -> Dict[str, Dict[str, List[str]]]:
    """The five derived split files, as filtered views of the master split."""
    subsets: Dict[str, Dict[str, List[str]]] = {
        key: {name: [] for name in SPLIT_NAMES} for key in FEATURE_KEYS
    }
    for name in SPLIT_NAMES:
        for doc_id in split.get(name, []):
            flags = document_features(documents[doc_id])
            for index, key in enumerate(FEATURE_KEYS):
                if flags[index]:
                    subsets[key][name].append(doc_id)
    return subsets


def split_corpus(
    raw_corpus: Mapping[str, Mapping[str, object]],
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 9876,
) -> Tuple[Dict[str, CleanDocument], Dict[str, List[str]]]:
    """Clean a raw corpus and produce its stratified master split."""
    cleaned = clean_corpus(raw_corpus)
    split = stratified_split(
        cleaned,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    return cleaned, split


# ---------------------------------------------------------------------------
# retrieval metrics
# ---------------------------------------------------------------------------


def rank_of_target(ranked: Sequence[str], target: str) -> int:
    """1-based rank of ``target`` in ``ranked``; 0 when absent."""
    for index, candidate in enumerate(ranked):
        if candidate == target:
            return index + 1
    return 0


def retrieval_metrics(
    ranked_lists: Sequence[Sequence[str]],
    targets: Sequence[str],
    *,
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """accuracy@k and mean reciprocal rank over Missing-Part predictions."""
    if len(ranked_lists) != len(targets):
        raise ValueError("ranked_lists and targets must have equal length")
    n = len(targets)
    if n == 0:
        return {"n": 0.0, "mrr": 0.0, **{f"acc@{k}": 0.0 for k in ks}}
    ranks = [rank_of_target(r, t) for r, t in zip(ranked_lists, targets)]
    out: Dict[str, float] = {"n": float(n)}
    for k in ks:
        out[f"acc@{k}"] = sum(1 for r in ranks if 0 < r <= k) / n
    out["mrr"] = sum((1.0 / r) if r else 0.0 for r in ranks) / n
    return out
