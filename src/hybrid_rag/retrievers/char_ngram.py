"""Character n-gram retriever, robust to typos and morphology.

Token-based retrievers miss "neural" when the query says "neurons", or fail on a
misspelled "photosynthesis". A character n-gram retriever represents each chunk as the
multiset of its character trigrams and scores by cosine similarity over those, so partial
and fuzzy matches still retrieve. It complements BM25 (exact tokens) and the TF-IDF dense
retriever (whole words) with a sub-word signal.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from hybrid_rag.core.types import Chunk, ScoredChunk

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def char_ngrams(text: str, n: int) -> Counter[str]:
    """Return the character n-gram multiset of ``text`` (lowercased, alnum runs)."""
    if n < 1:
        raise ValueError("n must be at least 1")
    normalized = _NON_ALNUM.sub(" ", text.lower()).strip()
    padded = f" {normalized} "
    return Counter(padded[i : i + n] for i in range(len(padded) - n + 1))


class CharNgramRetriever:
    """Retrieves by cosine similarity over character n-gram vectors."""

    def __init__(self, *, n: int = 3) -> None:
        if n < 1:
            raise ValueError("n must be at least 1")
        self._n = n
        self._chunks: list[Chunk] = []
        self._vectors: list[Counter[str]] = []
        self._norms: list[float] = []

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._vectors = [char_ngrams(chunk.text, self._n) for chunk in self._chunks]
        self._norms = [_norm(vector) for vector in self._vectors]

    def retrieve(self, query: str, *, k: int = 10) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        query_vector = char_ngrams(query, self._n)
        query_norm = _norm(query_vector)
        if query_norm == 0.0:
            return []
        scored: list[ScoredChunk] = []
        for index, vector in enumerate(self._vectors):
            similarity = _cosine(query_vector, query_norm, vector, self._norms[index])
            if similarity > 0.0:
                scored.append(ScoredChunk(chunk=self._chunks[index], score=similarity))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]


def _norm(vector: Counter[str]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def _cosine(a: Counter[str], a_norm: float, b: Counter[str], b_norm: float) -> float:
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(value * b.get(gram, 0) for gram, value in a.items())
    return dot / (a_norm * b_norm)
