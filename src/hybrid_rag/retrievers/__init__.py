"""Retrievers: sparse (BM25) and dense (TF-IDF), plus the retriever protocol."""

from __future__ import annotations

from hybrid_rag.core.retriever import Retriever
from hybrid_rag.retrievers.bm25 import BM25Retriever
from hybrid_rag.retrievers.dense import TfidfDenseRetriever

__all__ = [
    "BM25Retriever",
    "Retriever",
    "TfidfDenseRetriever",
]
