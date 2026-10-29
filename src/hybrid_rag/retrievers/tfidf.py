"""A TF-IDF vector-space retriever that fits its weights from the corpus.

This is a classical, sparse term-weighting retriever, not a neural dense embedding.
It fits inverse-document-frequency weights at index time, represents each chunk and
query as a sparse TF-IDF vector (a dictionary of term weights), and ranks by cosine
similarity. Fitting the idf from the indexed corpus is what makes it corpus-specific,
but the representation is sparse and lexical: it matches on shared terms and cannot
bridge paraphrases with no lexical overlap, which is what a neural dense retriever
adds. Implementing the fit, the transform, and cosine ranking from scratch shows the
mechanics with no heavy dependencies.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from hybrid_rag.core.tokenization import tokenize
from hybrid_rag.core.types import Chunk, ScoredChunk

Vector = dict[str, float]


class TfidfRetriever:
    """A retriever over sparse TF-IDF vectors ranked by cosine similarity."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._idf: dict[str, float] = {}
        self._vectors: list[Vector] = []
        self._norms: list[float] = []

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Fit the idf weights on ``chunks`` and embed each chunk."""
        self._chunks = list(chunks)
        self._fit_idf()
        self._vectors = [self._embed(tokenize(chunk.text)) for chunk in self._chunks]
        self._norms = [_norm(vector) for vector in self._vectors]

    def retrieve(self, query: str, *, k: int = 10) -> list[ScoredChunk]:
        """Return the top-``k`` chunks by cosine similarity to the query."""
        if not self._chunks:
            return []
        query_vector = self._embed(tokenize(query))
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

    def embedding_dimension(self) -> int:
        """The size of the learned vocabulary (the vector space dimension)."""
        return len(self._idf)

    def _fit_idf(self) -> None:
        document_frequency: Counter[str] = Counter()
        for chunk in self._chunks:
            document_frequency.update(set(tokenize(chunk.text)))
        n = len(self._chunks)
        # Smoothed idf, always positive, so every vocabulary term carries weight.
        self._idf = {
            term: math.log((n + 1) / (df + 1)) + 1.0 for term, df in document_frequency.items()
        }

    def _embed(self, tokens: Sequence[str]) -> Vector:
        if not tokens:
            return {}
        counts = Counter(tokens)
        length = len(tokens)
        vector: Vector = {}
        for term, count in counts.items():
            idf = self._idf.get(term)
            if idf is None:
                continue  # out-of-vocabulary terms do not contribute
            vector[term] = (count / length) * idf
        return vector


def _norm(vector: Vector) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def _cosine(a: Vector, a_norm: float, b: Vector, b_norm: float) -> float:
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    # Iterate over the smaller vector for efficiency.
    if len(a) > len(b):
        a, b = b, a
    dot = sum(value * b.get(term, 0.0) for term, value in a.items())
    return dot / (a_norm * b_norm)
