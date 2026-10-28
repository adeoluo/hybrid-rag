"""Hybrid RAG - production hybrid retrieval built from scratch.

Public API: the core types and chunking. Retrievers, fusion, reranking, and the
evaluation suite live in their own modules.
"""

from __future__ import annotations

from hybrid_rag.core import (
    Chunk,
    Document,
    FixedTokenChunker,
    ScoredChunk,
    SentenceChunker,
    chunk_documents,
    tokenize,
)

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "Document",
    "FixedTokenChunker",
    "ScoredChunk",
    "SentenceChunker",
    "__version__",
    "chunk_documents",
    "tokenize",
]
