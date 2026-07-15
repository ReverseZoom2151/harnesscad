"""similarity — stdlib-first similarity backends for memory recall.

The memory store's original retriever was a lexical Jaccard blend
(``TokenOverlapSimilarity`` in ``store.py``): symmetric set overlap plus a
difflib ratio. That is a *coordination* measure, not a *relevance* measure -- it
has no term saturation, no length normalisation and no notion that a rare word
("countersink") should count for more than a common one ("part"). This module
replaces it with two better backends that keep the same one-method ``Similarity``
seam (``score(query, doc) -> float`` in [0, 1]):

  * :class:`BM25Similarity` -- Okapi BM25, pure Python, the default. Term-
    frequency saturation (``k1``) and document-length normalisation (``b``) with
    a non-negative BM25+ IDF. It exposes both the pairwise ``score`` the protocol
    requires AND a corpus-aware :meth:`rank`; ``rank`` is where BM25 actually
    earns its keep, because IDF is only meaningful across a document set. Scores
    are normalised into [0, 1] by the query's own achievable BM25 mass so the
    existing ``min_similarity`` thresholds keep their meaning.

  * :class:`EmbeddingSimilarity` -- an OPTIONAL dense-cosine backend that
    activates ONLY when an embedding function is injected. It is never a hard
    dependency: no import of any model library happens here. Cosine is mapped
    from [-1, 1] to [0, 1] to honour the protocol's range contract.

Everything is deterministic and stdlib-only. Identical inputs yield identical
scores; there is no wall clock and no randomness.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence

__all__ = [
    "BM25Similarity",
    "EmbeddingSimilarity",
    "default_similarity",
    "make_similarity",
]

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall(str(text).lower())


def _bm25_idf(n_docs: int, doc_freq: int) -> float:
    """Non-negative BM25+ IDF: ln(1 + (N - df + 0.5) / (df + 0.5)).

    Always >= 0 (unlike the classic BM25 IDF, which can go negative for terms in
    more than half the corpus), so per-term contributions never subtract and the
    normalised score stays in [0, 1].
    """
    return math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))


class BM25Similarity:
    """Okapi BM25 similarity (pure Python), the default recall backend.

    Implements the :class:`~harnesscad.agents.memory.store.Similarity` protocol
    (``score(query, doc) -> float`` in [0, 1]) and adds a corpus-aware
    :meth:`rank`. ``k1`` controls term-frequency saturation, ``b`` controls
    document-length normalisation. An optional ``corpus`` (or a later call to
    :meth:`fit`) computes IDF and average document length across a real document
    set; without one, ``score`` degrades gracefully to a length-normalised,
    TF-saturated overlap that still strictly beats plain Jaccard.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        corpus: Optional[Sequence[str]] = None,
    ) -> None:
        if k1 < 0.0:
            raise ValueError("k1 must be >= 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self.k1 = float(k1)
        self.b = float(b)
        self._idf: Dict[str, float] = {}
        self._avgdl: float = 0.0
        self._n_docs: int = 0
        if corpus is not None:
            self.fit(corpus)

    # --- corpus statistics ------------------------------------------------
    def fit(self, corpus: Sequence[str]) -> "BM25Similarity":
        """Compute IDF and average document length over ``corpus`` (in place)."""
        docs = [_tokens(d) for d in corpus]
        self._n_docs = len(docs)
        total = sum(len(d) for d in docs)
        self._avgdl = (total / self._n_docs) if self._n_docs else 0.0
        df: Dict[str, int] = {}
        for d in docs:
            for term in set(d):
                df[term] = df.get(term, 0) + 1
        self._idf = {t: _bm25_idf(self._n_docs, c) for t, c in df.items()}
        return self

    @property
    def fitted(self) -> bool:
        return self._n_docs > 0

    def _idf_of(self, term: str) -> float:
        if term in self._idf:
            return self._idf[term]
        if not self.fitted:
            # No corpus: a single flat IDF for every term (N=1 smoothing). Ranking
            # then rests on TF saturation + length normalisation alone.
            return _bm25_idf(1, 1)
        # Fitted, but this query term is in no document: it cannot discriminate
        # among the docs, so it contributes nothing to the normalised score.
        return 0.0

    # --- scoring core -----------------------------------------------------
    def _score_tokens(
        self,
        q_tokens: Sequence[str],
        d_tokens: Sequence[str],
        avgdl: float,
        idf_of: Callable[[str], float],
    ) -> float:
        if not q_tokens or not d_tokens:
            return 0.0
        dl = len(d_tokens)
        tf = Counter(d_tokens)
        norm = 1.0 if avgdl <= 0.0 else dl / avgdl
        len_term = self.k1 * (1.0 - self.b + self.b * norm)
        raw = 0.0
        ref = 0.0
        for term in set(q_tokens):
            idf = idf_of(term)
            if idf <= 0.0:
                continue
            # The most a single term can contribute (as tf -> inf) is idf*(k1+1).
            ref += idf * (self.k1 + 1.0)
            f = tf.get(term, 0)
            if f:
                raw += idf * (f * (self.k1 + 1.0)) / (f + len_term)
        if ref <= 0.0:
            return 0.0
        s = raw / ref
        if s <= 0.0:
            return 0.0
        return 1.0 if s > 1.0 else s

    # --- Similarity protocol ---------------------------------------------
    def score(self, query: str, doc: str) -> float:
        """Pairwise BM25 in [0, 1] (1.0 iff ``doc`` saturates every query term).

        Uses the fitted corpus statistics when :meth:`fit` has been called,
        otherwise a self-contained single-document approximation.
        """
        q = _tokens(query)
        d = _tokens(doc)
        avgdl = self._avgdl if self.fitted else float(len(d))
        return self._score_tokens(q, d, avgdl, self._idf_of)

    # --- corpus-aware ranking --------------------------------------------
    def rank(self, query: str, docs: Sequence[str]) -> List[float]:
        """BM25 of ``query`` against every document in ``docs``, IDF computed
        over ``docs`` as the corpus. Returns one score in [0, 1] per document, in
        the same order as ``docs`` (empty list for empty input).

        This is the true Okapi ranking: a term common to every candidate gets a
        near-zero IDF and stops dominating, while a rare discriminating term is
        rewarded -- exactly the failure mode plain Jaccard could not express.
        """
        doc_tokens = [_tokens(d) for d in docs]
        n = len(doc_tokens)
        if n == 0:
            return []
        total = sum(len(t) for t in doc_tokens)
        avgdl = (total / n) if n else 0.0
        df: Dict[str, int] = {}
        for t in doc_tokens:
            for term in set(t):
                df[term] = df.get(term, 0) + 1
        idf_map = {term: _bm25_idf(n, c) for term, c in df.items()}

        def idf_of(term: str) -> float:
            # A query term absent from the corpus discriminates nothing here.
            return idf_map.get(term, 0.0)

        q = _tokens(query)
        return [self._score_tokens(q, t, avgdl, idf_of) for t in doc_tokens]


class EmbeddingSimilarity:
    """Optional dense-cosine similarity over an INJECTED embedding function.

    ``embed`` maps a string to a fixed-length vector (a list/tuple of floats):
    a sentence-transformer's ``encode``, an API embedder, or any deterministic
    callable. No model library is imported by this module -- the dependency, if
    any, lives entirely in the injected callable, so importing ``memory`` never
    pulls in a heavy embedding stack.

    Cosine similarity is mapped from [-1, 1] to [0, 1] to satisfy the
    ``Similarity`` protocol's range contract. Embeddings are cached by exact text
    so repeated recall over the same briefs does not re-embed.
    """

    def __init__(
        self,
        embed: Callable[[str], Sequence[float]],
        cache: bool = True,
    ) -> None:
        if not callable(embed):
            raise TypeError("embed must be a callable str -> sequence[float]")
        self._embed = embed
        self._cache_on = bool(cache)
        self._cache: Dict[str, List[float]] = {}

    def _vec(self, text: str) -> List[float]:
        if self._cache_on and text in self._cache:
            return self._cache[text]
        vec = [float(x) for x in self._embed(text)]
        if self._cache_on:
            self._cache[text] = vec
        return vec

    def score(self, query: str, doc: str) -> float:
        a = self._vec(query)
        b = self._vec(doc)
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        cos = dot / (math.sqrt(na) * math.sqrt(nb))
        # Map [-1, 1] -> [0, 1]; clamp for floating-point spill.
        s = (cos + 1.0) / 2.0
        if s <= 0.0:
            return 0.0
        return 1.0 if s > 1.0 else s


def default_similarity() -> BM25Similarity:
    """The default recall backend: Okapi BM25 with standard parameters."""
    return BM25Similarity()


def make_similarity(
    embed: Optional[Callable[[str], Sequence[float]]] = None,
) -> object:
    """Return an embedding-cosine backend if ``embed`` is given, else BM25.

    A one-call switch for callers (e.g. the pipeline) that may or may not have an
    embedder available: pass one to go dense, pass nothing to stay stdlib-only.
    """
    if embed is not None:
        return EmbeddingSimilarity(embed)
    return default_similarity()
