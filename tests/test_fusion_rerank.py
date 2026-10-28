"""Tests for hybrid fusion and reranking."""

from __future__ import annotations

import pytest

from hybrid_rag.core import Chunk
from hybrid_rag.reranking import (
    LexicalOverlapReranker,
    RerankedRetriever,
)
from hybrid_rag.retrievers import BM25Retriever, HybridRetriever, TfidfDenseRetriever


def _corpus() -> list[Chunk]:
    texts = {
        "c0": "The cat sat on the warm mat by the fire.",
        "c1": "Dogs are loyal animals that enjoy long walks.",
        "c2": "Python is a popular programming language for data science.",
        "c3": "The quick brown fox jumps over the lazy dog.",
        "c4": "Machine learning models are trained on large datasets in Python.",
    }
    return [Chunk(id=cid, document_id="d", text=text) for cid, text in texts.items()]


# --------------------------------------------------------------------------- #
# Hybrid fusion
# --------------------------------------------------------------------------- #
def test_hybrid_fuses_both_retrievers() -> None:
    hybrid = HybridRetriever([BM25Retriever(), TfidfDenseRetriever()])
    hybrid.index(_corpus())
    results = hybrid.retrieve("python programming", k=3)
    assert results
    assert results[0].chunk.id in {"c2", "c4"}


def test_hybrid_scores_descending() -> None:
    hybrid = HybridRetriever([BM25Retriever(), TfidfDenseRetriever()])
    hybrid.index(_corpus())
    scores = [r.score for r in hybrid.retrieve("dog", k=5)]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_requires_retrievers() -> None:
    with pytest.raises(ValueError, match="at least one"):
        HybridRetriever([])


def test_hybrid_validates_rrf_k() -> None:
    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetriever([BM25Retriever()], rrf_k=0)


def test_hybrid_single_retriever_matches_order() -> None:
    hybrid = HybridRetriever([BM25Retriever()])
    hybrid.index(_corpus())
    results = hybrid.retrieve("programming language python", k=1)
    assert results[0].chunk.id == "c2"


# --------------------------------------------------------------------------- #
# Reranking
# --------------------------------------------------------------------------- #
def test_reranker_reorders_candidates() -> None:
    retriever = BM25Retriever()
    retriever.index(_corpus())
    candidates = retriever.retrieve("python data science", k=5)
    reranked = LexicalOverlapReranker().rerank("python data science", candidates, k=3)
    assert reranked
    assert reranked[0].chunk.id == "c2"


def test_reranker_empty_query_returns_candidates() -> None:
    retriever = BM25Retriever()
    retriever.index(_corpus())
    candidates = retriever.retrieve("python", k=3)
    reranked = LexicalOverlapReranker().rerank("", candidates, k=2)
    assert len(reranked) == min(2, len(candidates))


def test_reranker_validates_weight() -> None:
    with pytest.raises(ValueError, match="coverage_weight"):
        LexicalOverlapReranker(coverage_weight=2.0)


def test_reranked_retriever_pipeline() -> None:
    pipeline = RerankedRetriever(
        HybridRetriever([BM25Retriever(), TfidfDenseRetriever()]),
        LexicalOverlapReranker(),
        candidate_pool=10,
    )
    pipeline.index(_corpus())
    results = pipeline.retrieve("machine learning datasets", k=2)
    assert results[0].chunk.id == "c4"
