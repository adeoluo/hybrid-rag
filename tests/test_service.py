"""Tests for the dependency-free HTTP retrieval service (pure handler layer)."""

from __future__ import annotations

from hybrid_rag.service import RetrievalService, handle

_SERVICE = RetrievalService.from_corpus()


def test_health() -> None:
    status, payload = handle("GET", "/health", {}, _SERVICE)
    assert status == 200
    assert payload["status"] == "ok"
    assert "hybrid" in payload["pipelines"]
    assert payload["chunks"] > 0


def test_retrieve_default_pipeline() -> None:
    status, payload = handle("POST", "/retrieve", {"query": "neural networks"}, _SERVICE)
    assert status == 200
    assert payload["pipeline"] == "hybrid"
    assert isinstance(payload["results"], list)


def test_retrieve_named_pipeline_and_k() -> None:
    status, payload = handle(
        "POST", "/retrieve", {"query": "photosynthesis", "pipeline": "bm25", "k": 3}, _SERVICE
    )
    assert status == 200
    assert len(payload["results"]) <= 3


def test_retrieve_rejects_missing_query() -> None:
    status, payload = handle("POST", "/retrieve", {}, _SERVICE)
    assert status == 400
    assert "query" in payload["error"]


def test_retrieve_rejects_unknown_pipeline() -> None:
    status, _ = handle("POST", "/retrieve", {"query": "x", "pipeline": "nope"}, _SERVICE)
    assert status == 400


def test_retrieve_rejects_bad_k() -> None:
    status, _ = handle("POST", "/retrieve", {"query": "x", "k": 0}, _SERVICE)
    assert status == 400


def test_unknown_route() -> None:
    status, payload = handle("GET", "/nope", {}, _SERVICE)
    assert status == 404
    assert "error" in payload
