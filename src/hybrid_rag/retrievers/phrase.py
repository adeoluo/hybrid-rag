"""Exact phrase retriever.

Some queries are about a specific phrase ("reciprocal rank fusion"), where token-bag
methods lose the ordering that makes the phrase meaningful. This retriever scores a chunk
by how many of the query's contiguous word n-grams appear verbatim in it, rewarding exact
phrase matches that BM25 and dense retrieval would under-rank.

Each distinct n-gram is indexed once per chunk with weight ``n``, so a query adds ``n`` for
every chunk containing each of its n-grams: the same integer score as intersecting the sets
chunk by chunk, computed only over chunks that share a phrase with the query.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hybrid_rag.core.postings import Postings, PostingsBuilder, top_k
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
        self._postings: Postings = Postings.empty()

    def _keys(self, tokens: Sequence[str]) -> list[tuple[str, float]]:
        # Tokens are alphanumeric runs, so joining with a space is unambiguous.
        return [
            (" ".join(gram), float(n))
            for n in range(1, self._max_n + 1)
            for gram in _ngrams(tokens, n)
        ]

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        builder = PostingsBuilder()
        for doc, chunk in enumerate(self._chunks):
            builder.add_chunk(doc, self._keys(tokenize(chunk.text, remove_stopwords=False)))
        self._postings = builder.build()

    def retrieve(self, query: str, *, k: int = 10) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        scores = np.zeros(len(self._chunks), dtype=np.float64)
        # Longer phrase matches are weighted more heavily (weight n for an n-gram).
        for key, weight in self._keys(tokenize(query, remove_stopwords=False)):
            posting = self._postings.get(key)
            if posting is not None:
                scores[posting[0]] += weight
        return [
            ScoredChunk(chunk=self._chunks[doc], score=score) for doc, score in top_k(scores, k)
        ]
