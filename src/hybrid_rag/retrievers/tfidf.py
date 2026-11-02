"""A TF-IDF vector-space retriever that fits its weights from the corpus.

This is a classical, sparse term-weighting retriever, not a neural dense embedding.
It fits inverse-document-frequency weights at index time, represents each chunk and
query as a sparse TF-IDF vector (a dictionary of term weights), and ranks by cosine
similarity. Fitting the idf from the indexed corpus is what makes it corpus-specific,
but the representation is sparse and lexical: it matches on shared terms and cannot
bridge paraphrases with no lexical overlap, which is what a neural dense retriever
adds. Implementing the fit, the transform, and cosine ranking from scratch keeps the
mechanics visible.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from hybrid_rag.core.postings import Postings, PostingsBuilder, Weights, top_k
from hybrid_rag.core.tokenization import tokenize
from hybrid_rag.core.types import Chunk, ScoredChunk

Vector = dict[str, float]


class TfidfRetriever:
    """A retriever over sparse TF-IDF vectors ranked by cosine similarity."""

    def __init__(self, *, stem: bool = True) -> None:
        self._stem = stem
        self._chunks: list[Chunk] = []
        self._postings: Postings = Postings.empty()
        self._idf: dict[str, float] = {}
        self._norms: Weights = np.zeros(0)

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Fit the idf weights on ``chunks`` and index each chunk's TF-IDF vector."""
        self._chunks = list(chunks)
        tokenized = [tokenize(chunk.text, stem=self._stem) for chunk in self._chunks]
        self._fit_idf(tokenized)
        builder = PostingsBuilder()
        norms: list[float] = []
        for doc, tokens in enumerate(tokenized):
            vector = self._embed(tokens)
            norms.append(_norm(vector))
            builder.add_chunk(doc, vector.items())
        # Only the postings and norms are kept; the per-chunk vectors are not needed again.
        self._postings = builder.build()
        self._norms = np.asarray(norms, dtype=np.float64)

    def retrieve(self, query: str, *, k: int = 10) -> list[ScoredChunk]:
        """Return the top-``k`` chunks by cosine similarity to the query."""
        if not self._chunks:
            return []
        query_vector = self._embed(tokenize(query, stem=self._stem))
        query_norm = _norm(query_vector)
        if query_norm == 0.0:
            return []
        # Accumulate dot products over the postings of the query's terms, so only chunks
        # sharing a term with the query are touched.
        dots = np.zeros(len(self._chunks), dtype=np.float64)
        for term, query_weight in query_vector.items():
            posting = self._postings.get(term)
            if posting is not None:
                ids, weights = posting
                dots[ids] += query_weight * weights
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.where(self._norms > 0.0, dots / (query_norm * self._norms), 0.0)
        return [
            ScoredChunk(chunk=self._chunks[doc], score=score) for doc, score in top_k(similarity, k)
        ]

    def embedding_dimension(self) -> int:
        """The size of the learned vocabulary (the vector space dimension)."""
        return len(self._idf)

    def _fit_idf(self, tokenized: Sequence[Sequence[str]]) -> None:
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))
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
