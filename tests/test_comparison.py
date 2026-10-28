"""Tests for additional metrics and retriever comparison."""

from __future__ import annotations

import pytest

from hybrid_rag.core import Chunk
from hybrid_rag.evaluation import (
    LabeledQuery,
    compare_retrievers,
    hit_rate,
    r_precision,
    success_at_k,
)
from hybrid_rag.pipelines import available_pipelines, build_pipeline
from hybrid_rag.retrievers import BM25Retriever, CharNgramRetriever


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_success_at_k() -> None:
    assert success_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
    assert success_at_k(["a", "b", "c"], {"z"}, 3) == 0.0


def test_hit_rate() -> None:
    assert hit_rate(["a", "b"], {"b"}) == 1.0
    assert hit_rate(["a", "b"], {"z"}) == 0.0


def test_r_precision() -> None:
    # 2 relevant; top-2 are a (relevant) and x (not): 1/2.
    assert r_precision(["a", "x", "b"], {"a", "b"}) == 0.5


def test_r_precision_empty() -> None:
    assert r_precision(["a"], set()) == 0.0


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def _corpus() -> list[Chunk]:
    texts = {
        "c0": "Photosynthesis converts sunlight into glucose in plants.",
        "c1": "Reciprocal rank fusion merges ranked result lists.",
        "c2": "Dogs are loyal animals that enjoy long walks.",
        "c3": "The neural network learned representations from data.",
    }
    return [Chunk(id=cid, document_id="d", text=text) for cid, text in texts.items()]


def _queries() -> list[LabeledQuery]:
    return [
        LabeledQuery(query="photosynthesis sunlight plants", relevant={"c0"}),
        LabeledQuery(query="reciprocal rank fusion", relevant={"c1"}),
        LabeledQuery(query="neural network data", relevant={"c3"}),
    ]


def test_compare_identical_retrievers_not_significant() -> None:
    corpus = _corpus()
    a = BM25Retriever()
    a.index(corpus)
    b = BM25Retriever()
    b.index(corpus)
    result = compare_retrievers(a, b, _queries(), metric="mrr")
    assert result.mean_a == result.mean_b
    assert not result.significant


def test_compare_reports_means() -> None:
    corpus = _corpus()
    a = BM25Retriever()
    a.index(corpus)
    b = CharNgramRetriever()
    b.index(corpus)
    result = compare_retrievers(a, b, _queries(), metric="success", k=3)
    assert 0.0 <= result.mean_a <= 1.0
    assert 0.0 <= result.mean_b <= 1.0
    assert 0.0 <= result.p_value <= 1.0


def test_compare_rejects_empty_queries() -> None:
    a = BM25Retriever()
    a.index(_corpus())
    with pytest.raises(ValueError, match="empty query set"):
        compare_retrievers(a, a, [], metric="mrr")


def test_compare_unknown_metric() -> None:
    a = BM25Retriever()
    a.index(_corpus())
    with pytest.raises(ValueError, match="unknown metric"):
        compare_retrievers(a, a, _queries(), metric="bogus")


# --------------------------------------------------------------------------- #
# New pipelines registered
# --------------------------------------------------------------------------- #
def test_new_pipelines_available() -> None:
    assert "hybrid-all" in available_pipelines()
    assert "mmr" in available_pipelines()


def test_new_pipelines_build_and_retrieve() -> None:
    corpus = _corpus()
    for name in ("hybrid-all", "mmr"):
        retriever = build_pipeline(name)
        retriever.index(corpus)
        results = retriever.retrieve("photosynthesis sunlight", k=2)
        assert results
