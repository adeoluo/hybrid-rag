"""Character n-gram retriever, robust to typos and morphology.

Token-based retrievers miss "neural" when the query says "neurons", or fail on a
misspelled "photosynthesis". A character n-gram retriever represents each chunk as the
multiset of its character trigrams and scores by cosine similarity over those, so partial
and fuzzy matches still retrieve. It complements BM25 (exact tokens) and the TF-IDF dense
retriever (whole words) with a sub-word signal.

Chunks are indexed by n-gram (see :mod:`hybrid_rag.core.postings`), so a query only touches
chunks sharing an n-gram with it. Dot products are sums of integer count products, which the
float accumulation adds exactly, so scores equal a direct cosine over every chunk.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

from hybrid_rag.core.postings import Postings, PostingsBuilder, Weights, top_k
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
        self._postings: Postings = Postings.empty()
        self._norms: Weights = np.zeros(0)

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        builder = PostingsBuilder()
        norms: list[float] = []
        for doc, chunk in enumerate(self._chunks):
            vector = char_ngrams(chunk.text, self._n)
            norms.append(_norm(vector))
            builder.add_chunk(doc, ((gram, float(count)) for gram, count in vector.items()))
        self._postings = builder.build()
        self._norms = np.asarray(norms, dtype=np.float64)

    def retrieve(self, query: str, *, k: int = 10) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        query_vector = char_ngrams(query, self._n)
        query_norm = _norm(query_vector)
        if query_norm == 0.0:
            return []
        dots = np.zeros(len(self._chunks), dtype=np.float64)
        for gram, count in query_vector.items():
            posting = self._postings.get(gram)
            if posting is not None:
                ids, counts = posting
                dots[ids] += count * counts
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.where(self._norms > 0.0, dots / (query_norm * self._norms), 0.0)
        return [
            ScoredChunk(chunk=self._chunks[doc], score=score) for doc, score in top_k(similarity, k)
        ]


def _norm(vector: Counter[str]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))
