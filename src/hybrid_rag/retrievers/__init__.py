"""Retrievers: sparse (BM25), dense (TF-IDF), and hybrid fusion."""

from __future__ import annotations

from hybrid_rag.core.retriever import Retriever
from hybrid_rag.retrievers.bm25 import BM25Retriever
from hybrid_rag.retrievers.char_ngram import CharNgramRetriever
from hybrid_rag.retrievers.dense import TfidfDenseRetriever
from hybrid_rag.retrievers.hybrid import HybridRetriever
from hybrid_rag.retrievers.phrase import PhraseRetriever

__all__ = [
    "BM25Retriever",
    "CharNgramRetriever",
    "HybridRetriever",
    "PhraseRetriever",
    "Retriever",
    "TfidfDenseRetriever",
]
