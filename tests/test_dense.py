"""Tests for the from-scratch neural dense retriever."""

from __future__ import annotations

from hybrid_rag.core import Chunk
from hybrid_rag.retrievers import DenseRetriever


def _corpus() -> list[Chunk]:
    texts = {
        "c0": "The cat sat on the warm mat by the fire in the house.",
        "c1": "Dogs are loyal animals that enjoy long walks outdoors.",
        "c2": "Python is a popular programming language for data science.",
        "c3": "The quick brown fox jumps over the lazy sleeping dog.",
        "c4": "Machine learning models are trained on large datasets of data.",
    }
    return [Chunk(id=cid, document_id="d", text=text) for cid, text in texts.items()]


def _retriever() -> DenseRetriever:
    return DenseRetriever(dim=16, epochs=20, seed=0)


def test_dense_indexes_and_retrieves() -> None:
    retriever = _retriever()
    retriever.index(_corpus())
    results = retriever.retrieve("programming language python", k=3)
    assert results
    assert len(results) == 3
    assert all(isinstance(r.chunk, Chunk) for r in results)


def test_dense_retrieves_exact_match_first() -> None:
    retriever = _retriever()
    corpus = _corpus()
    retriever.index(corpus)
    # Querying with a chunk's own text should rank that chunk first (mean-pooled
    # embedding of identical tokens is identical, so cosine is maximal).
    results = retriever.retrieve(corpus[2].text, k=5)
    assert results[0].chunk.id == "c2"


def test_dense_scores_are_descending() -> None:
    retriever = _retriever()
    retriever.index(_corpus())
    results = retriever.retrieve("dog walks outdoors", k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_dense_is_deterministic() -> None:
    a = _retriever()
    b = _retriever()
    corpus = _corpus()
    a.index(corpus)
    b.index(corpus)
    ra = [(r.chunk.id, round(r.score, 9)) for r in a.retrieve("data science", k=5)]
    rb = [(r.chunk.id, round(r.score, 9)) for r in b.retrieve("data science", k=5)]
    assert ra == rb


def test_dense_empty_index_returns_nothing() -> None:
    retriever = _retriever()
    assert retriever.retrieve("anything", k=5) == []


def test_dense_learns_embeddings() -> None:
    retriever = _retriever()
    retriever.index(_corpus())
    # Words that co-occur in the corpus should have an embedding; an unseen word
    # should not.
    assert retriever.embedding("python") is not None
    assert retriever.embedding("nonexistentword") is None
