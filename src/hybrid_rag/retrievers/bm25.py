"""BM25 lexical retriever, implemented from scratch.

BM25 (Okapi BM25) is the strong, standard sparse-retrieval baseline. It scores a chunk for
a query by summing, over the query terms, an inverse-document-frequency weight times a
saturating term-frequency component that is normalized by chunk length. Two parameters
tune it: ``k1`` (term-frequency saturation) and ``b`` (length normalization).

Implementing it directly - rather than importing a library - makes the ranking function
transparent and is the point of the exercise.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from hybrid_rag.core.tokenization import tokenize
from hybrid_rag.core.types import Chunk, ScoredChunk


class BM25Retriever:
    """A from-scratch Okapi BM25 retriever."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self._k1 = k1
        self._b = b
        self._chunks: list[Chunk] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_freqs: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._idf: dict[str, float] = {}
        self._avg_len = 0.0

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Build the inverted-statistics index over ``chunks``."""
        self._chunks = list(chunks)
        self._doc_tokens = [tokenize(chunk.text) for chunk in self._chunks]
        self._doc_freqs = [Counter(tokens) for tokens in self._doc_tokens]
        self._df = Counter()
        for freqs in self._doc_freqs:
            self._df.update(freqs.keys())
        total_len = sum(len(tokens) for tokens in self._doc_tokens)
        self._avg_len = total_len / len(self._doc_tokens) if self._doc_tokens else 0.0
        self._idf = {term: self._compute_idf(df) for term, df in self._df.items()}

    def retrieve(self, query: str, *, k: int = 10) -> list[ScoredChunk]:
        """Return the top-``k`` chunks by BM25 score."""
        if not self._chunks:
            return []
        query_terms = tokenize(query)
        scored = [
            ScoredChunk(chunk=self._chunks[i], score=self._score(query_terms, i))
            for i in range(len(self._chunks))
        ]
        scored = [item for item in scored if item.score > 0.0]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]

    def _score(self, query_terms: Sequence[str], doc_index: int) -> float:
        freqs = self._doc_freqs[doc_index]
        doc_len = len(self._doc_tokens[doc_index])
        score = 0.0
        for term in query_terms:
            if term not in freqs:
                continue
            tf = freqs[term]
            idf = self._idf.get(term, 0.0)
            denominator = tf + self._k1 * (1 - self._b + self._b * doc_len / self._avg_len)
            score += idf * tf * (self._k1 + 1) / denominator
        return score

    def _compute_idf(self, df: int) -> float:
        # Okapi BM25 idf with the standard +0.5 smoothing; clamped at zero so terms in
        # more than half the corpus never contribute a negative score.
        n = len(self._chunks)
        return max(0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0))
