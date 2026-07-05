"""Hybrid retriever — fuse BM25 + dense-ish vector ranks (blueprint sec.7, sec.19 P2).

Lexical (BM25) and dense-ish (hashed-vector cosine) retrieval fail in different
ways: BM25 nails exact terms but misses paraphrase/overlap; cosine smooths over
vocabulary but can be dragged off by generically-similar prose. Fusing their
*rankings* recovers the on-topic chunk that each ranks second — the hybrid
routinely beats either index alone on mixed queries.

Fusion default is **Reciprocal Rank Fusion** (RRF): scale-free, no score
normalisation needed, robust to BM25's unbounded scores vs cosine's [0, 1]. A
weighted-score mode is available for when both indexes are well-calibrated.

``build_from_docs`` is the convenience entry point: hand it file paths, raw
texts, or ``(text, source)`` pairs and it chunks + indexes everything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from rag.chunk import Chunk, chunk_document
from rag.index import BM25Index, Embedder, HashedVectorIndex


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
    ) -> None:
        if fusion not in ("rrf", "weighted"):
            raise ValueError("fusion must be 'rrf' or 'weighted'")
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.bm25 = BM25Index()
        self.vector = HashedVectorIndex(embedder=embedder)
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
