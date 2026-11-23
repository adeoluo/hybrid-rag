"""The offline ablation figures quoted in the README and the evaluation report are current."""

from __future__ import annotations

from pathlib import Path

from hybrid_rag.cli import _CHUNKER
from hybrid_rag.core.chunking import chunk_documents
from hybrid_rag.data.sample import corpus, labeled_queries
from hybrid_rag.evaluation.harness import evaluate_retriever
from hybrid_rag.pipelines import build_pipeline

ROOT = Path(__file__).resolve().parent.parent


def _report(name: str, k: int = 5):  # type: ignore[no-untyped-def]
    retriever = build_pipeline(name)
    retriever.index(chunk_documents(corpus(), _CHUNKER))
    return evaluate_retriever(retriever, labeled_queries(), k=k)


def test_quoted_ablation_figures_are_what_the_command_prints() -> None:
    bm25, mmr = _report("bm25"), _report("mmr")
    readme = (ROOT / "README.md").read_text()
    report = (ROOT / "docs" / "EVALUATION_REPORT.md").read_text()
    for text in (readme, report):
        assert f"`recall@5 = {bm25.recall:.3f}`" in text
        assert f"`MRR = {bm25.mrr:.3f}`" in text
    assert f"`{mmr.mrr:.3f}` for `mmr`" in readme
    assert f"`nDCG@5 = {bm25.ndcg:.3f}`" in report
    assert f"`MRR = {mmr.mrr:.3f}`, `nDCG@5 = {mmr.ndcg:.3f}`" in report
    assert f"{len(corpus())}-document / {len(labeled_queries())}-query" in report
