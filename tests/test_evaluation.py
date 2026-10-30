"""Tests for retrieval metrics and the evaluation harness."""

from __future__ import annotations

import pytest

from hybrid_rag.core import Chunk
from hybrid_rag.evaluation import (
    LabeledQuery,
    average_precision,
    evaluate_retriever,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from hybrid_rag.retrievers import BM25Retriever, HybridRetriever, TfidfRetriever


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_recall_at_k() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "d"}, 3) == 0.5
    assert recall_at_k(["a", "b"], {"a", "b"}, 2) == 1.0


def test_recall_empty_relevant() -> None:
    assert recall_at_k(["a"], set(), 1) == 0.0


def test_precision_at_k() -> None:
    assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, 4) == 0.5


def test_precision_at_k_denominator_is_k() -> None:
    # Regression: dividing by the number of results returned inflated precision
    # when fewer than k items were retrieved. One correct result at k=5 is 0.2.
    assert precision_at_k(["a"], {"a"}, 5) == pytest.approx(0.2)
    assert precision_at_k(["a", "b", "c"], {"a"}, 5) == pytest.approx(0.2)
    assert precision_at_k([], {"a"}, 5) == 0.0


def test_precision_validates_k() -> None:
    with pytest.raises(ValueError, match="k must"):
        precision_at_k(["a"], {"a"}, 0)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], {"a"}) == 1.0
    assert reciprocal_rank(["x"], {"a"}) == 0.0


def test_average_precision() -> None:
    # Relevant at ranks 1 and 3: (1/1 + 2/3) / 2.
    ap = average_precision(["a", "x", "b"], {"a", "b"})
    assert ap == pytest.approx((1.0 + 2 / 3) / 2)


def test_ndcg_perfect_and_imperfect() -> None:
    gains = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert ndcg_at_k(["a", "b", "c"], gains, 3) == pytest.approx(1.0)
    worse = ndcg_at_k(["c", "b", "a"], gains, 3)
    assert worse < 1.0


def test_ndcg_empty_gains() -> None:
    assert ndcg_at_k(["a"], {}, 1) == 0.0


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def _corpus() -> list[Chunk]:
    texts = {
        "c0": "The cat sat on the warm mat by the fire.",
        "c1": "Dogs are loyal animals that enjoy long walks.",
        "c2": "Python is a popular programming language for data science.",
        "c3": "The quick brown fox jumps over the lazy dog.",
        "c4": "Machine learning models are trained on large datasets in Python.",
    }
    return [Chunk(id=cid, document_id="d", text=text) for cid, text in texts.items()]


def _queries() -> list[LabeledQuery]:
    return [
        LabeledQuery(query="python programming language", relevant={"c2"}),
        LabeledQuery(query="machine learning datasets", relevant={"c4"}),
        LabeledQuery(query="loyal dogs walks", relevant={"c1"}),
    ]


def test_evaluate_bm25() -> None:
    retriever = BM25Retriever()
    retriever.index(_corpus())
    report = evaluate_retriever(retriever, _queries(), k=3)
    assert report.num_queries == 3
    assert 0.0 <= report.recall <= 1.0
    assert report.mrr > 0.0


def test_evaluate_rejects_empty_queries() -> None:
    retriever = BM25Retriever()
    retriever.index(_corpus())
    with pytest.raises(ValueError, match="empty query set"):
        evaluate_retriever(retriever, [], k=3)


def test_report_format() -> None:
    retriever = BM25Retriever()
    retriever.index(_corpus())
    report = evaluate_retriever(retriever, _queries(), k=3)
    text = report.format()
    assert "recall@3" in text
    assert "MRR" in text
    assert "nDCG@3" in text


def test_hybrid_at_least_matches_components_on_mrr() -> None:
    corpus = _corpus()
    queries = _queries()

    bm25 = BM25Retriever()
    bm25.index(corpus)
    bm25_report = evaluate_retriever(bm25, queries, k=3)

    hybrid = HybridRetriever([BM25Retriever(), TfidfRetriever()])
    hybrid.index(corpus)
    hybrid_report = evaluate_retriever(hybrid, queries, k=3)

    # On this small corpus both find the single relevant chunk; hybrid should not be worse.
    assert hybrid_report.mrr >= bm25_report.mrr - 1e-9
