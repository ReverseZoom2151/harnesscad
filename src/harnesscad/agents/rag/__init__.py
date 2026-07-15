"""rag — the hybrid RAG grounding layer (blueprint sec.2, sec.7, sec.19 Phase 2).

Grounds generation in standards / API docs via **hybrid retrieval**: structure-
aware chunking (headings kept as breadcrumbs, code/API blocks atomic) feeds a
classic BM25 lexical index and an embedding-free hashed-vector cosine index,
fused by reciprocal-rank fusion in a ``HybridRetriever``.

STDLIB ONLY — no external embedding model or vector DB. Same lightweight,
pluggable-similarity philosophy as ``memory/store.py``: an ``Embedder`` protocol
lets a real model swap in later without touching call sites.

    from harnesscad.agents.rag import HybridRetriever, build_from_docs
    r = build_from_docs(["standards/iso4762.md", ("raw text", "notes")])
    hits = r.retrieve("M6 socket head cap screw torque", k=5)
"""

from __future__ import annotations

from harnesscad.agents.rag.chunk import Chunk, chunk_document, chunk_documents
from harnesscad.agents.rag.index import (
    BM25Index,
    Embedder,
    HashedEmbedder,
    HashedVectorIndex,
    Index,
    tokenize,
)
from harnesscad.agents.rag.retriever import (
    HybridRetriever,
    Retrieved,
    SimilarityBM25Index,
    SimilarityEmbedder,
    build_from_docs,
)
from harnesscad.agents.rag.retrieval_eval import (
    EvalCase,
    RetrievalReport,
    evaluate,
    mrr,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "Chunk",
    "chunk_document",
    "chunk_documents",
    "BM25Index",
    "HashedVectorIndex",
    "HashedEmbedder",
    "Embedder",
    "Index",
    "tokenize",
    "HybridRetriever",
    "Retrieved",
    "SimilarityBM25Index",
    "SimilarityEmbedder",
    "build_from_docs",
    "EvalCase",
    "RetrievalReport",
    "evaluate",
    "recall_at_k",
    "reciprocal_rank",
    "mrr",
]
