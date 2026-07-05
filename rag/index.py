"""Retrieval indexes for the hybrid RAG layer (blueprint sec.7, sec.19 Phase 2).

Two complementary indexes behind one ``Index`` interface:

  - **``BM25Index``** — classic Okapi BM25. Sparse lexical retrieval: rewards
    rare query terms (idf), saturates on term frequency (k1), normalises by
    document length (b). This is the workhorse for exact standard/part numbers,
    tolerances, API symbol names — the vocabulary an engineer types verbatim.

  - **``HashedVectorIndex``** — an *embedding-free* dense-ish index. It hashes
    token n-grams into a fixed-width vector and ranks by cosine. No model, no
    vector DB, deterministic — yet it captures lexical *overlap smoothing*
    (shared bigrams, morphology) that plain BM25 misses, giving the fusion layer
    a genuinely different signal to combine. A pluggable ``Embedder`` protocol
    lets a real embedder (sentence-transformers, an API embedder) drop straight
    in later without touching the index or the retriever.

STDLIB ONLY — BM25 and the hashed embedder are implemented here from scratch,
in the same lightweight-similarity spirit as ``memory/store.py``.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Dict, List, Protocol, Sequence, Tuple

from rag.chunk import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenisation (shared by every index)."""
    return _TOKEN.findall(text.lower())


def _stable_hash(s: str) -> int:
    """Process-stable hash (Python's ``hash`` on str is salted; md5 is not)."""
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "big")


# ---------------------------------------------------------------------------
# Common interface
# ---------------------------------------------------------------------------
class Index(Protocol):
    """Common index interface: ``add(chunk)`` then ``search(query, k)``.

    ``search`` returns ``[(chunk, score), ...]`` sorted by score descending.
    Scores are index-local (BM25 is unbounded, cosine is in [0, 1]); the
    retriever fuses them by *rank*, so absolute scales need not agree.
    """

    def add(self, chunk: Chunk) -> None: ...

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]: ...


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
class BM25Index:
    """Okapi BM25 over chunk text (stdlib, incremental).

    Corpus statistics (df, avgdl) are kept up to date on every ``add`` so
    ``search`` is a straight scan. Fine for the standards/API-doc corpus sizes
    this grounding layer targets; swap in an inverted-index scan if it grows.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.tfs: List[Dict[str, int]] = []       # per-doc term frequencies
        self.lengths: List[int] = []              # per-doc token counts
        self.df: Dict[str, int] = {}              # document frequency per term

    def add(self, chunk: Chunk) -> None:
        toks = tokenize(chunk.text)
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        self.chunks.append(chunk)
        self.tfs.append(tf)
        self.lengths.append(len(toks))
        for term in tf:
            self.df[term] = self.df.get(term, 0) + 1

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self.df.get(term, 0)
        # Okapi idf with +1 floor so common terms never go negative.
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_index: int) -> float:
        if not self.chunks:
            return 0.0
        avgdl = sum(self.lengths) / len(self.lengths)
        tf = self.tfs[doc_index]
        dl = self.lengths[doc_index]
        s = 0.0
        for term in tokenize(query):
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = self._idf(term)
            denom = f + self.k1 * (1 - self.b + self.b * dl / avgdl)
            s += idf * (f * (self.k1 + 1)) / denom
        return s

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        scored: List[Tuple[float, int]] = []
        for i in range(len(self.chunks)):
            sc = self.score(query, i)
            if sc > 0.0:
                scored.append((sc, i))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [(self.chunks[i], sc) for sc, i in scored[:k]]


# ---------------------------------------------------------------------------
# Embedder protocol + embedding-free default
# ---------------------------------------------------------------------------
class Embedder(Protocol):
    """Pluggable text -> vector strategy.

    The default is embedding-free (hashed n-grams). A real embedder implements
    the same one-method interface — ``embed(text) -> sparse dict[int, float]`` —
    and swaps in via ``HashedVectorIndex(embedder=MyEmbedder())`` with no other
    change. Sparse dict form keeps the interface agnostic to dimensionality.
    """

    def embed(self, text: str) -> Dict[int, float]: ...


class HashedEmbedder:
    """Embedding-free embedder: hashed token n-grams -> sparse term vector.

    Unigrams give lexical presence; higher n-grams give a little word-order and
    phrase sensitivity, so cosine here behaves differently from bag-of-words
    BM25 — which is exactly what makes hybrid fusion worth doing. ``dim`` bounds
    the hashing space (collisions are rare and harmless at these corpus sizes).
    """

    def __init__(self, dim: int = 4096, ngrams: Sequence[int] = (1, 2)) -> None:
        self.dim = dim
        self.ngrams = tuple(ngrams)

    def embed(self, text: str) -> Dict[int, float]:
        toks = tokenize(text)
        vec: Dict[int, float] = {}
        for n in self.ngrams:
            if n <= 0:
                continue
            for i in range(len(toks) - n + 1):
                gram = " ".join(toks[i:i + n])
                idx = _stable_hash(f"{n}:{gram}") % self.dim
                vec[idx] = vec.get(idx, 0.0) + 1.0
        return vec


def _cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    # Iterate the smaller vector for the dot product.
    if len(a) > len(b):
        a, b = b, a
    dot = sum(w * b.get(i, 0.0) for i, w in a.items())
    if dot == 0.0:
        return 0.0
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class HashedVectorIndex:
    """Dense-ish cosine index over hashed n-gram vectors (embedding-free)."""

    def __init__(self, embedder: Embedder = None) -> None:
        self.embedder: Embedder = embedder or HashedEmbedder()
        self.chunks: List[Chunk] = []
        self.vectors: List[Dict[int, float]] = []

    def add(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)
        self.vectors.append(self.embedder.embed(chunk.text))

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        qv = self.embedder.embed(query)
        scored: List[Tuple[float, int]] = []
        for i, dv in enumerate(self.vectors):
            sc = _cosine(qv, dv)
            if sc > 0.0:
                scored.append((sc, i))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [(self.chunks[i], sc) for sc, i in scored[:k]]
