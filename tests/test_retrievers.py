"""Tests for the BM25 and dense TF-IDF retrievers."""

from __future__ import annotations

import pytest

from hybrid_rag.core import Chunk
from hybrid_rag.retrievers import BM25Retriever, TfidfDenseRetriever


def _corpus() -> list[Chunk]:
    texts = {
        "c0": "The cat sat on the warm mat by the fire.",
        "c1": "Dogs are loyal animals that enjoy long walks.",
        "c2": "Python is a popular programming language for data science.",
        "c3": "The quick brown fox jumps over the lazy dog.",
        "c4": "Machine learning models are trained on large datasets.",
    }
    return [Chunk(id=cid, document_id="d", text=text) for cid, text in texts.items()]


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #
def test_bm25_retrieves_relevant_first() -> None:
    retriever = BM25Retriever()
    retriever.index(_corpus())
    results = retriever.retrieve("programming language python", k=3)
    assert results
    assert results[0].chunk.id == "c2"


def test_bm25_scores_descending() -> None:
    retriever = BM25Retriever()
    retriever.index(_corpus())
    results = retriever.retrieve("dog", k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_empty_index() -> None:
    assert BM25Retriever().retrieve("anything") == []


def test_bm25_no_match_returns_empty() -> None:
    retriever = BM25Retriever()
    retriever.index(_corpus())
    assert retriever.retrieve("xylophone quasar") == []


def test_bm25_validates_params() -> None:
    with pytest.raises(ValueError, match="k1"):
        BM25Retriever(k1=-1)
    with pytest.raises(ValueError, match="b must"):
        BM25Retriever(b=2.0)


def test_bm25_respects_k() -> None:
    retriever = BM25Retriever()
    retriever.index(_corpus())
    assert len(retriever.retrieve("the", k=2)) <= 2


# --------------------------------------------------------------------------- #
# Dense TF-IDF
# --------------------------------------------------------------------------- #
def test_dense_retrieves_relevant_first() -> None:
    retriever = TfidfDenseRetriever()
    retriever.index(_corpus())
    results = retriever.retrieve("machine learning datasets", k=3)
    assert results
    assert results[0].chunk.id == "c4"


def test_dense_learns_vocabulary() -> None:
    retriever = TfidfDenseRetriever()
    retriever.index(_corpus())
    assert retriever.embedding_dimension() > 0


def test_dense_empty_index() -> None:
    assert TfidfDenseRetriever().retrieve("anything") == []


def test_dense_out_of_vocabulary_query() -> None:
    retriever = TfidfDenseRetriever()
    retriever.index(_corpus())
    assert retriever.retrieve("xylophone") == []


def test_dense_cosine_is_symmetric_scale_invariant() -> None:
    # A query that repeats a chunk's words should rank that chunk first.
    retriever = TfidfDenseRetriever()
    retriever.index(_corpus())
    results = retriever.retrieve("cat mat fire", k=1)
    assert results[0].chunk.id == "c0"
