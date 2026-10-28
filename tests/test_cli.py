"""Tests for the bundled dataset, pipelines, and CLI."""

from __future__ import annotations

import pytest

from hybrid_rag.cli import main
from hybrid_rag.core.chunking import FixedTokenChunker, chunk_documents
from hybrid_rag.data.sample import corpus, labeled_queries
from hybrid_rag.evaluation.harness import evaluate_retriever
from hybrid_rag.pipelines import available_pipelines, build_pipeline


def test_corpus_and_queries_consistent() -> None:
    docs = corpus()
    queries = labeled_queries()
    chunk_ids = {chunk.id for chunk in chunk_documents(docs, FixedTokenChunker(chunk_size=200))}
    for query in queries:
        # Every relevant id must correspond to a real chunk.
        assert query.relevant <= chunk_ids


def test_all_pipelines_build() -> None:
    assert set(available_pipelines()) == {"bm25", "dense", "hybrid", "reranked"}
    for name in available_pipelines():
        assert build_pipeline(name) is not None


def test_build_unknown_pipeline_raises() -> None:
    with pytest.raises(KeyError, match="Unknown pipeline"):
        build_pipeline("nonexistent")


def test_pipelines_achieve_strong_retrieval() -> None:
    chunks = chunk_documents(corpus(), FixedTokenChunker(chunk_size=200))
    queries = labeled_queries()
    for name in available_pipelines():
        retriever = build_pipeline(name)
        retriever.index(chunks)
        report = evaluate_retriever(retriever, queries, k=3)
        # On the clean corpus every pipeline should reach high recall and MRR.
        assert report.recall >= 0.9
        assert report.mrr >= 0.9


def test_cli_ablate(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["ablate", "--k", "3"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "bm25" in output
    assert "reranked" in output
    assert "recall" in output


def test_cli_query(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["query", "how do plants make energy", "--pipeline", "hybrid", "--k", "2"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "photosynthesis" in output


def test_cli_query_dense(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["query", "powerhouse of the cell", "--pipeline", "dense"])
    assert exit_code == 0
    assert "mitochondria" in capsys.readouterr().out
