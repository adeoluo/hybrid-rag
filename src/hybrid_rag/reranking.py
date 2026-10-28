"""Reranking: a second-stage scorer over a first-stage candidate set.

First-stage retrievers optimize recall cheaply over the whole corpus; a reranker then
re-scores the small candidate set more precisely to improve the ordering at the top, which
is what a downstream reader actually consumes. This is the standard retrieve-then-rerank
architecture.

:class:`LexicalOverlapReranker` is a transparent, offline cross-encoder-style scorer: it
scores each candidate by the overlap between the query's and the chunk's term sets,
weighted toward covering more of the query. A neural cross-encoder would slot in here
unchanged, behind the same :class:`Reranker` interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from hybrid_rag.core.retriever import Retriever
from hybrid_rag.core.tokenization import tokenize
from hybrid_rag.core.types import Chunk, ScoredChunk


@runtime_checkable
class Reranker(Protocol):
    """Re-scores a candidate set of chunks for a query."""

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], *, k: int = 10
    ) -> list[ScoredChunk]:
        """Return up to ``k`` candidates re-scored and re-ordered."""
        ...


class LexicalOverlapReranker:
    """A cross-encoder-style reranker scoring query/chunk term overlap.

    The score combines query coverage (what fraction of query terms appear in the chunk)
    with a mild precision term (how concentrated the chunk is on query terms), which
    rewards chunks that both cover the query and are not padded with unrelated text.
    """

    def __init__(self, *, coverage_weight: float = 0.8) -> None:
        if not 0.0 <= coverage_weight <= 1.0:
            raise ValueError("coverage_weight must be in [0, 1]")
        self._coverage_weight = coverage_weight

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], *, k: int = 10
    ) -> list[ScoredChunk]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return list(candidates[:k])
        rescored = [
            ScoredChunk(chunk=candidate.chunk, score=self._score(query_terms, candidate))
            for candidate in candidates
        ]
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:k]

    def _score(self, query_terms: set[str], candidate: ScoredChunk) -> float:
        chunk_terms = set(tokenize(candidate.chunk.text))
        if not chunk_terms:
            return 0.0
        overlap = query_terms & chunk_terms
        coverage = len(overlap) / len(query_terms)
        precision = len(overlap) / len(chunk_terms)
        return self._coverage_weight * coverage + (1 - self._coverage_weight) * precision


class RerankedRetriever:
    """Composes a first-stage retriever with a reranker into one Retriever.

    It retrieves a deeper candidate pool, reranks it, and returns the top-k - so the whole
    retrieve-then-rerank pipeline satisfies the plain Retriever interface and can be
    evaluated like any other retriever.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        *,
        candidate_pool: int = 40,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._candidate_pool = candidate_pool

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._retriever.index(chunks)

    def retrieve(self, query: str, *, k: int = 10) -> list[ScoredChunk]:
        candidates = self._retriever.retrieve(query, k=self._candidate_pool)
        return self._reranker.rerank(query, candidates, k=k)
