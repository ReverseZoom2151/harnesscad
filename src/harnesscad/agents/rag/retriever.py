"""Hybrid retriever — fuse BM25 + dense-ish vector ranks (blueprint sec.7, sec.19 P2).

Lexical (BM25) and dense-ish (hashed-vector cosine) retrieval fail in different
ways: BM25 nails exact terms but misses paraphrase/overlap; cosine smooths over
vocabulary but can be dragged off by generically-similar prose. Fusing their
*rankings* recovers the on-topic chunk that each ranks second — the hybrid
routinely beats either index alone on mixed queries.

Fusion default is **Reciprocal Rank Fusion** (RRF): scale-free, no score
normalisation needed, robust to BM25's unbounded scores vs cosine's [0, 1]. A
weighted-score mode is available for when both indexes are well-calibrated.

SPARSE + DENSE, NOT JUST TOKEN OVERLAP (blueprint sec.16.3). Both signals reuse
``harnesscad.agents.memory.similarity`` when it is importable, so the RAG and
memory layers score relevance identically:

  - the SPARSE side prefers ``similarity.BM25Similarity`` (Okapi BM25+, corpus-
    aware ``rank``, [0, 1]-normalised) over the local ``index.BM25Index``; if the
    similarity module is absent it degrades to that local BM25;
  - the DENSE side stays embedding-free (hashed n-gram cosine) by default, but a
    real embedding function can be injected (``embed_fn=``) and is wrapped into
    the vector index -- the same "inject an embedder, no hard dependency" seam
    ``similarity.EmbeddingSimilarity`` uses.

The similarity module is imported LAZILY and every reuse is guarded: a missing or
broken backend never fails a retrieval, it just falls back to the stdlib path.

``build_from_docs`` is the convenience entry point: hand it file paths, raw
texts, or ``(text, source)`` pairs and it chunks + indexes everything.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from harnesscad.agents.rag.chunk import Chunk, chunk_document
from harnesscad.agents.rag.index import BM25Index, Embedder, HashedVectorIndex


# ---------------------------------------------------------------------------
# Sparse reuse: agents.memory.similarity.BM25Similarity (lazy, optional)
# ---------------------------------------------------------------------------
class SimilarityBM25Index:
    """BM25 index backed by ``agents.memory.similarity.BM25Similarity``.

    Exposes the same ``add(chunk)`` / ``search(query, k) -> [(chunk, score)]``
    interface as ``index.BM25Index`` so it drops straight into the fusion, but
    ranks with the memory layer's Okapi BM25+ (non-negative IDF, [0, 1]-
    normalised, IDF recomputed over the live candidate set on each search). Using
    it means the RAG and memory layers agree on what "lexically relevant" means.
    """

    def __init__(self, backend: Any = None) -> None:
        if backend is None:
            from harnesscad.agents.memory.similarity import BM25Similarity
            backend = BM25Similarity()
        self._sim = backend
        self.chunks: List[Chunk] = []

    def add(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        if not self.chunks:
            return []
        scores = self._sim.rank(query, [c.text for c in self.chunks])
        scored = [(float(s), i) for i, s in enumerate(scores) if s > 0.0]
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [(self.chunks[i], s) for s, i in scored[:k]]


def _load_similarity_bm25() -> Optional[SimilarityBM25Index]:
    """Build a similarity-backed BM25 index, or None if the module is absent.

    Imported lazily and guarded: a missing memory.similarity module (or an
    import-time failure) returns None so the caller degrades to local BM25.
    """
    try:
        importlib.import_module("harnesscad.agents.memory.similarity")
        return SimilarityBM25Index()
    except Exception:  # noqa: BLE001 - module absent / import failure -> degrade
        return None


def _resolve_bm25(bm25: str) -> Tuple[Any, str]:
    """Choose the sparse index: 'auto' | 'similarity' | 'local'.

    'auto' and 'similarity' reuse memory.similarity's BM25 when importable;
    'auto' silently falls back to the local BM25, 'similarity' also falls back
    (reuse is a preference, never a hard requirement). Returns (index, kind).
    """
    if bm25 not in ("auto", "similarity", "local"):
        raise ValueError("bm25 must be 'auto', 'similarity', or 'local'")
    if bm25 != "local":
        sim = _load_similarity_bm25()
        if sim is not None:
            return sim, "similarity"
    return BM25Index(), "local"


# ---------------------------------------------------------------------------
# Dense reuse: an injected embedding function (the EmbeddingSimilarity seam)
# ---------------------------------------------------------------------------
def _to_sparse(vec: Any) -> Dict[int, float]:
    """Project a dense vector (or sparse mapping) into the ``Dict[int, float]``
    form the hashed-vector cosine consumes. Unusable inputs collapse to ``{}``."""
    if vec is None:
        return {}
    if isinstance(vec, dict):
        out: Dict[int, float] = {}
        for k, v in vec.items():
            try:
                out[int(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return out
    try:
        return {i: float(v) for i, v in enumerate(vec) if float(v) != 0.0}
    except (TypeError, ValueError):
        return {}


class SimilarityEmbedder:
    """Adapt a raw ``text -> vector`` embedding function to the ``Embedder``
    protocol (``embed(text) -> Dict[int, float]``).

    This is the same injection seam ``similarity.EmbeddingSimilarity`` uses: no
    model library is imported here, the dependency lives entirely in the injected
    callable. The dense vector it returns is projected into the sparse form the
    ``HashedVectorIndex`` cosine already ranks by, so a real embedder drops into
    the existing fusion without touching the index. A failing embed call degrades
    to an empty vector rather than raising.
    """

    def __init__(self, fn: Callable[[str], Any]) -> None:
        if not callable(fn):
            raise TypeError("embed_fn must be callable str -> sequence[float]")
        self._fn = fn

    def embed(self, text: str) -> Dict[int, float]:
        try:
            vec = self._fn(text)
        except Exception:  # noqa: BLE001 - a broken embedder degrades to no vector
            return {}
        return _to_sparse(vec)


def _resolve_embedder(embedder: Embedder,
                      embed_fn: Optional[Callable[[str], Any]]
                      ) -> Tuple[Optional[Embedder], str]:
    """Pick the dense embedder and report its kind.

    - an explicit ``embedder`` wins (kind ``"custom"``);
    - an ``embed_fn`` is wrapped into a :class:`SimilarityEmbedder` (``"embed_fn"``);
    - otherwise ``None`` lets ``HashedVectorIndex`` build its default embedding-
      free ``HashedEmbedder`` (``"hashed"``), preserving the prior behaviour.
    """
    if embedder is not None:
        return embedder, "custom"
    if embed_fn is not None:
        return SimilarityEmbedder(embed_fn), "embed_fn"
    return None, "hashed"


@dataclass
class Retrieved:
    """A ranked result: the chunk plus its fused score and per-index ranks."""

    chunk: Chunk
    score: float
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None

    # Convenience passthroughs so callers can treat this like a chunk.
    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def heading_path(self) -> List[str]:
        return self.chunk.heading_path


ChunkFilter = Callable[[Chunk], bool]
DocSpec = Union[str, Tuple[str, str], Chunk]


class HybridRetriever:
    """BM25 + hashed-vector retrieval with rank fusion and optional filtering."""

    def __init__(
        self,
        embedder: Embedder = None,
        fusion: str = "rrf",
        rrf_k: int = 60,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        bm25: str = "auto",
        embed_fn: Optional[Callable[[str], Any]] = None,
    ) -> None:
        if fusion not in ("rrf", "weighted"):
            raise ValueError("fusion must be 'rrf' or 'weighted'")
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        # SPARSE. 'auto' reuses memory.similarity's BM25 when importable, else the
        # local BM25 -- either way the same add/search interface feeds the fusion.
        self.bm25, self.bm25_kind = _resolve_bm25(bm25)
        # DENSE. Embedding-free hashed cosine by default; an injected embed_fn (or
        # an explicit embedder) upgrades the vector index to real embeddings.
        resolved, self.embedder_kind = _resolve_embedder(embedder, embed_fn)
        self.vector = HashedVectorIndex(embedder=resolved)
        self.chunks: List[Chunk] = []

    # --- ingestion --------------------------------------------------------
    def add_chunk(self, chunk: Chunk) -> None:
        self.bm25.add(chunk)
        self.vector.add(chunk)
        self.chunks.append(chunk)

    def add_document(self, text: str, source: str = "doc") -> List[Chunk]:
        cs = chunk_document(text, source)
        for c in cs:
            self.add_chunk(c)
        return cs

    def build_from_docs(self, docs: Sequence[DocSpec]) -> "HybridRetriever":
        """Chunk + index a mixed list of file paths, raw texts, or pairs/chunks.

        Each item may be:
          - a ``Chunk``               -> indexed as-is (already chunked),
          - a ``(text, source)`` pair -> chunked with that source label,
          - a ``str`` that is an existing file path -> read + chunked,
          - any other ``str``         -> treated as raw document text.
        """
        for i, item in enumerate(docs):
            if isinstance(item, Chunk):
                self.add_chunk(item)
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                text, source = item
                self.add_document(text, source)
            elif isinstance(item, str):
                if os.path.exists(item) and os.path.isfile(item):
                    with open(item, "r", encoding="utf-8") as fh:
                        self.add_document(fh.read(), os.path.basename(item))
                else:
                    self.add_document(item, f"doc{i}")
            else:
                raise TypeError(f"unsupported doc spec: {type(item).__name__}")
        return self

    # --- retrieval --------------------------------------------------------
    def retrieve(
        self,
        query: str,
        k: int = 5,
        source: Optional[str] = None,
        heading: Optional[str] = None,
        where: Optional[ChunkFilter] = None,
        candidate_pool: Optional[int] = None,
    ) -> List[Retrieved]:
        """Return the top-``k`` fused results, most-relevant first.

        Optional filters (applied before fusion so ``k`` is honoured against the
        filtered set):
          - ``source``  : substring match on ``chunk.source``.
          - ``heading`` : substring match on any breadcrumb in ``heading_path``.
          - ``where``   : arbitrary predicate over the chunk.
        """
        if not self.chunks:
            return []
        pool = candidate_pool if candidate_pool is not None else max(k * 5, 20)

        keep = self._make_filter(source, heading, where)

        bm = [(c, s) for (c, s) in self.bm25.search(query, pool) if keep(c)]
        vc = [(c, s) for (c, s) in self.vector.search(query, pool) if keep(c)]

        if self.fusion == "rrf":
            fused = self._fuse_rrf(bm, vc)
        else:
            fused = self._fuse_weighted(bm, vc)

        fused.sort(key=lambda r: (-r.score, r.chunk.ordinal, r.chunk.source))
        return fused[:k]

    def retrieve_chunks(self, query: str, k: int = 5, **kw) -> List[Chunk]:
        """Convenience: same as ``retrieve`` but returns bare ``Chunk`` objects."""
        return [r.chunk for r in self.retrieve(query, k, **kw)]

    # --- fusion helpers ---------------------------------------------------
    def _make_filter(self, source, heading, where) -> ChunkFilter:
        def keep(c: Chunk) -> bool:
            if source is not None and source not in c.source:
                return False
            if heading is not None and not any(heading in h for h in c.heading_path):
                return False
            if where is not None and not where(c):
                return False
            return True
        return keep

    def _fuse_rrf(self, bm, vc) -> List[Retrieved]:
        ranks_bm = {c.id: i for i, (c, _s) in enumerate(bm)}
        ranks_vc = {c.id: i for i, (c, _s) in enumerate(vc)}
        by_id: Dict[str, Chunk] = {}
        for c, _s in bm:
            by_id[c.id] = c
        for c, _s in vc:
            by_id[c.id] = c

        out: List[Retrieved] = []
        for cid, chunk in by_id.items():
            score = 0.0
            rb = ranks_bm.get(cid)
            rv = ranks_vc.get(cid)
            if rb is not None:
                score += 1.0 / (self.rrf_k + rb + 1)
            if rv is not None:
                score += 1.0 / (self.rrf_k + rv + 1)
            out.append(Retrieved(chunk=chunk, score=score, bm25_rank=rb, vector_rank=rv))
        return out

    def _fuse_weighted(self, bm, vc) -> List[Retrieved]:
        # Min-max normalise each index's scores to [0, 1] before weighting.
        def norm(pairs) -> Dict[str, float]:
            if not pairs:
                return {}
            scores = [s for _c, s in pairs]
            lo, hi = min(scores), max(scores)
            span = hi - lo
            if span <= 0:
                return {c.id: 1.0 for c, _s in pairs}
            return {c.id: (s - lo) / span for c, s in pairs}

        nbm = norm(bm)
        nvc = norm(vc)
        ranks_bm = {c.id: i for i, (c, _s) in enumerate(bm)}
        ranks_vc = {c.id: i for i, (c, _s) in enumerate(vc)}
        by_id: Dict[str, Chunk] = {}
        for c, _s in bm:
            by_id[c.id] = c
        for c, _s in vc:
            by_id[c.id] = c

        out: List[Retrieved] = []
        for cid, chunk in by_id.items():
            score = (self.bm25_weight * nbm.get(cid, 0.0)
                     + self.vector_weight * nvc.get(cid, 0.0))
            out.append(Retrieved(
                chunk=chunk, score=score,
                bm25_rank=ranks_bm.get(cid), vector_rank=ranks_vc.get(cid),
            ))
        return out


def build_from_docs(docs: Sequence[DocSpec], embedder: Embedder = None,
                    **kw) -> HybridRetriever:
    """Module-level convenience: build a ready-to-query ``HybridRetriever``."""
    return HybridRetriever(embedder=embedder, **kw).build_from_docs(docs)
