"""Maximal Marginal Relevance (MMR) diversity reranking.

A retriever that returns five near-duplicate chunks wastes the context window. MMR
re-ranks candidates to balance relevance to the query against novelty relative to the
already-selected results, greedily picking the candidate that maximizes

    lambda * relevance - (1 - lambda) * max_similarity_to_selected

Relevance reuses each candidate's first-stage score; similarity between chunks is Jaccard
over their token sets. It satisfies the :class:`~hybrid_rag.reranking.Reranker` protocol,
so it drops into the retrieve-then-rerank pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence

from hybrid_rag.core.tokenization import tokenize
from hybrid_rag.core.types import ScoredChunk


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


class MMRReranker:
    """Re-ranks candidates to trade off relevance against diversity."""

    def __init__(self, *, diversity: float = 0.3) -> None:
        """Create an MMR reranker.

        Args:
            diversity: Weight on novelty in ``[0, 1]``. ``0`` is pure relevance (no
                diversity); ``1`` is pure novelty. The MMR lambda is ``1 - diversity``.
        """
        if not 0.0 <= diversity <= 1.0:
            raise ValueError("diversity must be in [0, 1]")
        self._lambda = 1.0 - diversity

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], *, k: int = 10
    ) -> list[ScoredChunk]:
        pool = list(candidates)
        if not pool:
            return []
        token_sets = {c.id: set(tokenize(c.chunk.text)) for c in pool}
        # Normalize relevance to [0, 1] so it is comparable to the similarity term.
        max_score = max(c.score for c in pool) or 1.0

        selected: list[ScoredChunk] = []
        remaining = pool.copy()
        while remaining and len(selected) < k:
            best = max(
                remaining,
                key=lambda c: self._mmr_value(c, selected, token_sets, max_score),
            )
            selected.append(best)
            remaining.remove(best)
        return selected

    def _mmr_value(
        self,
        candidate: ScoredChunk,
        selected: list[ScoredChunk],
        token_sets: dict[str, set[str]],
        max_score: float,
    ) -> float:
        relevance = candidate.score / max_score
        if not selected:
            return self._lambda * relevance
        max_similarity = max(_jaccard(token_sets[candidate.id], token_sets[s.id]) for s in selected)
        return self._lambda * relevance - (1.0 - self._lambda) * max_similarity
