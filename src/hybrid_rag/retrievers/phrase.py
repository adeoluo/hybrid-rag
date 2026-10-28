"""Exact phrase retriever.

Some queries are about a specific phrase ("reciprocal rank fusion"), where token-bag
methods lose the ordering that makes the phrase meaningful. This retriever scores a chunk
by how many of the query's contiguous word n-grams appear verbatim in it, rewarding exact
phrase matches that BM25 and dense retrieval would under-rank.
"""

from __future__ import annotations

from collections.abc import Sequence

from hybrid_rag.core.tokenization import tokenize
from hybrid_rag.core.types import Chunk, ScoredChunk


def _ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


class PhraseRetriever:
    """Scores chunks by contiguous word-n-gram overlap with the query."""

    def __init__(self, *, max_n: int = 3) -> None:
        if max_n < 1:
            raise ValueError("max_n must be at least 1")
        self._max_n = max_n
        self._chunks: list[Chunk] = []
        self._chunk_grams: list[dict[int, set[tuple[str, ...]]]] = []

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._chunk_grams = []
        for chunk in self._chunks:
            tokens = tokenize(chunk.text, remove_stopwords=False)
            self._chunk_grams.append({n: _ngrams(tokens, n) for n in range(1, self._max_n + 1)})

    def retrieve(self, query: str, *, k: int = 10) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        query_tokens = tokenize(query, remove_stopwords=False)
        query_grams = {n: _ngrams(query_tokens, n) for n in range(1, self._max_n + 1)}
        scored: list[ScoredChunk] = []
        for index, chunk in enumerate(self._chunks):
            score = 0.0
            for n in range(1, self._max_n + 1):
                overlap = len(query_grams[n] & self._chunk_grams[index][n])
                # Longer phrase matches are weighted more heavily.
                score += overlap * n
            if score > 0.0:
                scored.append(ScoredChunk(chunk=chunk, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]
