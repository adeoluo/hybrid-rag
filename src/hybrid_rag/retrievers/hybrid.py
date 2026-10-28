"""Hybrid retrieval via reciprocal rank fusion.

Sparse (BM25) and dense (TF-IDF) retrievers fail on different queries: BM25 rewards exact
term matches, the dense retriever rewards distributional similarity. Combining them
usually beats either alone. The challenge is that their scores are on incomparable scales,
so they cannot simply be added.

Reciprocal rank fusion (RRF) sidesteps this by combining *ranks*, not scores: each chunk's
fused score is the sum over retrievers of ``1 / (k + rank)``. It is parameter-light,
robust, and the standard way to merge ranked lists.
"""

from __future__ import annotations

from collections.abc import Sequence

from hybrid_rag.core.retriever import Retriever
from hybrid_rag.core.types import Chunk, ScoredChunk


class HybridRetriever:
    """Fuses several retrievers' rankings with reciprocal rank fusion."""

    def __init__(self, retrievers: Sequence[Retriever], *, rrf_k: int = 60) -> None:
        """Create a hybrid retriever.

        Args:
            retrievers: The component retrievers to fuse.
            rrf_k: The RRF constant; larger values flatten the contribution of top ranks.
        """
        if not retrievers:
            raise ValueError("HybridRetriever requires at least one retriever")
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        self._retrievers = tuple(retrievers)
        self._rrf_k = rrf_k

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Index ``chunks`` in every component retriever."""
        for retriever in self._retrievers:
            retriever.index(chunks)

    def retrieve(self, query: str, *, k: int = 10) -> list[ScoredChunk]:
        """Return the top-``k`` chunks by fused reciprocal rank."""
        # Retrieve a deeper pool from each component so fusion has material to work with.
        pool = max(k * 4, 20)
        fused: dict[str, float] = {}
        chunk_by_id: dict[str, Chunk] = {}
        for retriever in self._retrievers:
            results = retriever.retrieve(query, k=pool)
            for rank, scored in enumerate(results):
                chunk_by_id[scored.id] = scored.chunk
                fused[scored.id] = fused.get(scored.id, 0.0) + 1.0 / (self._rrf_k + rank + 1)
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        return [
            ScoredChunk(chunk=chunk_by_id[chunk_id], score=score) for chunk_id, score in ranked[:k]
        ]
