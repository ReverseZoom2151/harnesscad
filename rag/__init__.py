"""rag — the hybrid RAG grounding layer (blueprint sec.2, sec.7, sec.19 Phase 2).

Grounds generation in standards / API docs via **hybrid retrieval**: structure-
aware chunking (headings kept as breadcrumbs, code/API blocks atomic) feeds a
classic BM25 lexical index and an embedding-free hashed-vector cosine index,
fused by reciprocal-rank fusion in a ``HybridRetriever``.

STDLIB ONLY — no external embedding model or vector DB. Same lightweight,
pluggable-similarity philosophy as ``memory/store.py``: an ``Embedder`` protocol
lets a real model swap in later without touching call sites.

    from rag import HybridRetriever, build_from_docs
    r = build_from_docs(["standards/iso4762.md", ("raw text", "notes")])
    hits = r.retrieve("M6 socket head cap screw torque", k=5)
"""

from __future__ import annotations

from rag.chunk import Chunk, chunk_document, chunk_documents
from rag.index import (
    BM25Index,
    Embedder,
    HashedEmbedder,
    HashedVectorIndex,
    Index,
    tokenize,
)
from rag.retriever import HybridRetriever, Retrieved, build_from_docs

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
    "build_from_docs",
]
