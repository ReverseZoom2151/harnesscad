"""The "Two Parts" benchmark: do these two part names belong to one assembly?

Reimplementation of ``generate_pairs.py`` from "What's In A Name?" (Meltzer,
Lambourne, Grandi, JCISE 2023), plus the evaluation metrics the paper reports.

Protocol (fully deterministic, seeded ``random.Random`` -- no numpy, no nltk):

1. For every document with >= 2 user-authored part names, shuffle its parts with
   the seeded RNG, take the first part, and pair it with the first *later* part
   whose token set is **disjoint** from the first's. The disjointness rule is
   what makes the task semantic rather than lexical: "left wheel" / "right
   wheel" would be trivially solvable by string overlap, so it is rejected.
   A document contributes at most one positive pair.
2. Build a global co-occurrence table part -> {parts seen in the same document},
   over *all* splits, so a negative pair cannot accidentally be a true pair
   observed elsewhere in the corpus.
3. Negative pairs: resample the ``b`` column of the positive pairs and keep
   ``(a, b')`` only when ``b'`` never co-occurs with ``a``. Positives are then
   truncated so the two classes are balanced, and the emitted rows are all
   positives first, then all negatives (the CSV layout the paper documents).

Metrics: ``pair_accuracy`` at a threshold, the best-threshold sweep, and a
tie-aware ``roc_auc`` computed from ranks (no sklearn).
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from harnesscad.domain.library.name_normalizer import is_user_name, tokenize_name


@dataclass(frozen=True)
class NamePair:
    """One benchmark row: ``label`` is 1 for same-assembly, 0 otherwise."""

    label: int
    a: str
    b: str

    def as_row(self) -> Tuple[int, str, str]:
        return (self.label, self.a, self.b)


def _user_parts(doc: Mapping[str, object]) -> List[str]:
    return [str(n) for n in (doc.get("body_names") or []) if is_user_name(str(n))]


def cooccurrence_table(
    corpus: Mapping[str, Mapping[str, object]]
) -> Dict[str, Set[str]]:
    """Map each part name to the set of parts it shares a document with."""
    table: Dict[str, Set[str]] = {}
    for doc_id in sorted(corpus):
        parts = _user_parts(corpus[doc_id])
        unique = sorted(set(parts))
        for part in unique:
            others = set(unique)
            others.discard(part)
            table.setdefault(part, set()).update(others)
    return table


def positive_pairs(
    corpus: Mapping[str, Mapping[str, object]],
    document_ids: Sequence[str],
    *,
    seed: int = 9876,
) -> List[Tuple[str, str]]:
    """Token-disjoint same-document pairs, at most one per document."""
    rng = random.Random(seed)
    pairs: List[Tuple[str, str]] = []
    for doc_id in document_ids:
        parts = _user_parts(corpus.get(doc_id, {}))
        if len(parts) < 2:
            continue
        shuffled = rng.sample(parts, k=len(parts))
        first, rest = shuffled[0], shuffled[1:]
        first_tokens = set(tokenize_name(first))
        for other in rest:
            if first_tokens.isdisjoint(set(tokenize_name(other))):
                pairs.append((first, other))
                break
    return pairs


def negative_pairs(
    positives: Sequence[Tuple[str, str]],
    table: Mapping[str, Set[str]],
    *,
    seed: int = 9876,
) -> List[Tuple[str, str]]:
    """Resample the ``b`` column; keep only never-co-occurring ``(a, b')``."""
    if not positives:
        return []
    rng = random.Random(seed)
    a_col = [a for a, _ in positives]
    b_col = [b for _, b in positives]
    shuffled_b = rng.sample(b_col, k=len(b_col))
    negatives: List[Tuple[str, str]] = []
    for a, b in zip(a_col, shuffled_b):
        if a == b:
            continue
        if b in table.get(a, set()):
            continue
        negatives.append((a, b))
    return negatives


def build_pairs(
    corpus: Mapping[str, Mapping[str, object]],
    document_ids: Sequence[str],
    *,
    seed: int = 9876,
    balance: bool = True,
) -> List[NamePair]:
    """Full protocol: positives, then negatives, class-balanced by truncation."""
    table = cooccurrence_table(corpus)
    pos = positive_pairs(corpus, document_ids, seed=seed)
    neg = negative_pairs(pos, table, seed=seed)
    if balance and len(neg) < len(pos):
        pos = pos[: len(neg)]
    rows = [NamePair(1, a, b) for a, b in pos]
    rows.extend(NamePair(0, a, b) for a, b in neg)
    return rows


def pairs_to_csv(pairs: Iterable[NamePair]) -> str:
    """Headerless ``label,a,b`` CSV exactly as the released dataset stores it."""
    lines = []
    for pair in pairs:
        a = pair.a.replace('"', '""')
        b = pair.b.replace('"', '""')
        lines.append(f'{pair.label},"{a}","{b}"')
    return "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def pair_accuracy(
    labels: Sequence[int], scores: Sequence[float], *, threshold: float = 0.5
) -> float:
    """Fraction correct when predicting positive iff ``score >= threshold``."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length")
    if not labels:
        return 0.0
    correct = sum(
        1
        for label, score in zip(labels, scores)
        if (1 if score >= threshold else 0) == int(label)
    )
    return correct / len(labels)


def best_threshold(
    labels: Sequence[int], scores: Sequence[float]
) -> Tuple[float, float]:
    """Return ``(threshold, accuracy)`` maximising accuracy over candidate cuts.

    Ties are broken toward the smaller threshold, so the result is deterministic.
    """
    if not labels:
        return (0.0, 0.0)
    candidates = sorted(set(scores))
    best = (candidates[0], -1.0)
    for cut in candidates:
        acc = pair_accuracy(labels, scores, threshold=cut)
        if acc > best[1]:
            best = (cut, acc)
    return best


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Tie-aware ROC AUC via the rank (Mann-Whitney U) identity."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length")
    n_pos = sum(1 for label in labels if int(label) == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based, averaged over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_sum = sum(ranks[i] for i, label in enumerate(labels) if int(label) == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate_pairs(
    pairs: Sequence[NamePair], scores: Sequence[float], *, threshold: float = 0.5
) -> Dict[str, float]:
    """Accuracy at ``threshold``, best-threshold accuracy, AUC and class counts."""
    labels = [p.label for p in pairs]
    cut, best_acc = best_threshold(labels, scores)
    return {
        "n": float(len(pairs)),
        "positives": float(sum(labels)),
        "negatives": float(len(labels) - sum(labels)),
        "accuracy": pair_accuracy(labels, scores, threshold=threshold),
        "best_threshold": cut,
        "best_accuracy": best_acc,
        "roc_auc": roc_auc(labels, scores),
    }
