"""Training-free PPMI baseline for assembly-part name semantics.

"What's In A Name?" (Meltzer, Lambourne, Grandi, JCISE 2023) evaluates learned
encoders (BOW, FastText, TechNet, DistilBERT, DistilBERT-FT) on three tasks:
Two-Parts (do these names co-occur in one assembly?), Missing-Part, and
Document-Name. Every one of those encoders needs a trained model, so none of
them is reproducible inside this harness.

This module supplies the deterministic, closed-form baseline the paper lacks: a
count-based **PPMI (positive pointwise mutual information)** vector space built
from the part-name corpus itself. Nothing is trained -- the vectors are a
deterministic function of the counts:

    PPMI(t, c) = max(0, log2( p(t, c) / (p(t) * p(c)) ))

Token-token co-occurrence is taken over the parts of a single assembly (a token
in part A co-occurs with every token of every other part in the same document),
which is exactly the signal the Two-Parts task probes. A part name is embedded
as the mean of its token vectors (the paper's own pooling for FastText), and
pairs are scored by cosine similarity.

Also provided: an IDF-weighted BOW/TF-IDF cosine scorer (the paper's "Frequency
BOW" / "TF-IDF BOW" ablations) as a lexical control -- expected to score near
chance on the benchmark, because pairs are constructed token-disjoint.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from harnesscad.domain.library.name_normalizer import is_user_name, tokenize_name


def _parts_of(doc: Mapping[str, object]) -> List[str]:
    return [str(n) for n in (doc.get("body_names") or []) if is_user_name(str(n))]


@dataclass(frozen=True)
class PPMIModel:
    """A sparse PPMI vector space over part-name tokens."""

    vectors: Dict[str, Dict[str, float]]
    norms: Dict[str, float]
    idf: Dict[str, float]

    @property
    def vocabulary(self) -> List[str]:
        return sorted(self.vectors)

    def token_vector(self, token: str) -> Dict[str, float]:
        return dict(self.vectors.get(token, {}))

    # -- embedding -------------------------------------------------------

    def name_vector(self, name: str) -> Dict[str, float]:
        """Mean of the PPMI vectors of a name's known tokens."""
        tokens = [t for t in tokenize_name(name) if t in self.vectors]
        if not tokens:
            return {}
        acc: Dict[str, float] = {}
        for token in tokens:
            for context, weight in self.vectors[token].items():
                acc[context] = acc.get(context, 0.0) + weight
        scale = 1.0 / len(tokens)
        return {c: w * scale for c, w in acc.items()}

    def set_vector(self, names: Sequence[str]) -> Dict[str, float]:
        """Mean of the name vectors of a part set (the Set-Transformer input)."""
        vecs = [v for v in (self.name_vector(n) for n in names) if v]
        if not vecs:
            return {}
        acc: Dict[str, float] = {}
        for vec in vecs:
            for context, weight in vec.items():
                acc[context] = acc.get(context, 0.0) + weight
        scale = 1.0 / len(vecs)
        return {c: w * scale for c, w in acc.items()}

    # -- scoring ---------------------------------------------------------

    def pair_score(self, a: str, b: str) -> float:
        """Cosine similarity of two part names in PPMI space (0 if unknown)."""
        return cosine(self.name_vector(a), self.name_vector(b))

    def score_pairs(self, pairs: Iterable[Tuple[str, str]]) -> List[float]:
        return [self.pair_score(a, b) for a, b in pairs]

    def rank_candidates(
        self, inputs: Sequence[str], candidates: Sequence[str]
    ) -> List[str]:
        """Rank candidate names by similarity to a part set (Missing-Part).

        Ties break alphabetically, so the ranking is deterministic.
        """
        query = self.set_vector(inputs)
        scored = [
            (-cosine(query, self.name_vector(c)), c) for c in candidates
        ]
        scored.sort()
        return [name for _score, name in scored]

    # -- lexical control -------------------------------------------------

    def tfidf_vector(self, name: str) -> Dict[str, float]:
        counts = Counter(tokenize_name(name))
        if not counts:
            return {}
        total = sum(counts.values())
        return {
            token: (count / total) * self.idf.get(token, 0.0)
            for token, count in counts.items()
        }

    def tfidf_pair_score(self, a: str, b: str) -> float:
        return cosine(self.tfidf_vector(a), self.tfidf_vector(b))


def cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Cosine similarity of two sparse vectors; 0.0 when either is empty."""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(weight * b.get(key, 0.0) for key, weight in a.items())
    if dot == 0.0:
        return 0.0
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def token_cooccurrence(
    corpus: Mapping[str, Mapping[str, object]],
    document_ids: Sequence[str],
) -> Tuple[Counter, Counter, int]:
    """Token-token co-occurrence across distinct parts of the same document.

    Returns ``(pair_counts, token_counts, total)`` where a pair is an ordered
    ``(t, c)`` tuple counted symmetrically.
    """
    pair_counts: Counter = Counter()
    token_counts: Counter = Counter()
    total = 0
    for doc_id in document_ids:
        parts = _parts_of(corpus.get(doc_id, {}))
        token_sets = [sorted(set(tokenize_name(p))) for p in parts]
        for i, left in enumerate(token_sets):
            for j, right in enumerate(token_sets):
                if i == j:
                    continue
                for t in left:
                    for c in right:
                        if t == c:
                            continue
                        pair_counts[(t, c)] += 1
                        token_counts[t] += 1
                        total += 1
    return pair_counts, token_counts, total


def build_ppmi(
    corpus: Mapping[str, Mapping[str, object]],
    document_ids: Sequence[str],
    *,
    min_count: int = 1,
    shift: float = 0.0,
) -> PPMIModel:
    """Build the PPMI space from the training documents only.

    ``shift`` implements shifted-PPMI (SPPMI): ``max(0, pmi - shift)``, which
    prunes weak associations exactly as a negative-sampling objective would --
    but in closed form.
    """
    pair_counts, token_counts, total = token_cooccurrence(corpus, document_ids)
    vectors: Dict[str, Dict[str, float]] = {}
    if total > 0:
        for (t, c), count in pair_counts.items():
            if count < min_count:
                continue
            p_tc = count / total
            p_t = token_counts[t] / total
            p_c = token_counts[c] / total
            if p_t <= 0.0 or p_c <= 0.0:
                continue
            pmi = math.log2(p_tc / (p_t * p_c)) - shift
            if pmi > 0.0:
                vectors.setdefault(t, {})[c] = pmi

    norms = {
        token: math.sqrt(sum(w * w for w in vec.values()))
        for token, vec in vectors.items()
    }

    # document-frequency IDF over part names (for the lexical control)
    df: Counter = Counter()
    n_names = 0
    for doc_id in document_ids:
        for part in _parts_of(corpus.get(doc_id, {})):
            tokens = set(tokenize_name(part))
            if not tokens:
                continue
            n_names += 1
            for token in tokens:
                df[token] += 1
    idf = {
        token: math.log((1 + n_names) / (1 + count)) + 1.0
        for token, count in df.items()
    }
    return PPMIModel(vectors=vectors, norms=norms, idf=idf)


def nearest_tokens(model: PPMIModel, token: str, *, k: int = 5) -> List[Tuple[str, float]]:
    """The ``k`` tokens most similar to ``token`` in PPMI space."""
    query = model.token_vector(token)
    if not query:
        return []
    scored = [
        (-cosine(query, model.vectors[other]), other)
        for other in model.vectors
        if other != token
    ]
    scored.sort()
    return [(name, -score) for score, name in scored[:k]]
