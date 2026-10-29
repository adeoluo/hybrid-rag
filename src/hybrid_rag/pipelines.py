"""Convenience builders for the named retriever configurations.

Four configurations are compared throughout the project: BM25 alone, the dense retriever
alone, their hybrid fusion, and the hybrid pipeline followed by reranking. These builders
give each a stable name so the CLI and the evaluation report refer to the same things.
"""

from __future__ import annotations

from collections.abc import Callable

from hybrid_rag.core.retriever import Retriever
from hybrid_rag.mmr import MMRReranker
from hybrid_rag.reranking import LexicalOverlapReranker, RerankedRetriever
from hybrid_rag.retrievers import (
    BM25Retriever,
    CharNgramRetriever,
    HybridRetriever,
    PhraseRetriever,
    TfidfRetriever,
)


def _hybrid() -> Retriever:
    return HybridRetriever([BM25Retriever(), TfidfRetriever()])


def _reranked() -> Retriever:
    return RerankedRetriever(_hybrid(), LexicalOverlapReranker(), candidate_pool=20)


def _hybrid_all() -> Retriever:
    return HybridRetriever(
        [BM25Retriever(), TfidfRetriever(), CharNgramRetriever(), PhraseRetriever()]
    )


def _mmr() -> Retriever:
    return RerankedRetriever(_hybrid(), MMRReranker(diversity=0.3), candidate_pool=20)


BUILDERS: dict[str, Callable[[], Retriever]] = {
    "bm25": BM25Retriever,
    "tfidf": TfidfRetriever,
    "hybrid": _hybrid,
    "reranked": _reranked,
    "hybrid-all": _hybrid_all,
    "mmr": _mmr,
}


def available_pipelines() -> list[str]:
    """Return the names of the built-in retriever configurations."""
    return list(BUILDERS)


def build_pipeline(name: str) -> Retriever:
    """Construct a named retriever configuration.

    Raises:
        KeyError: If ``name`` is not a known configuration.
    """
    try:
        return BUILDERS[name]()
    except KeyError:
        raise KeyError(
            f"Unknown pipeline {name!r}. Available: {', '.join(available_pipelines())}"
        ) from None
