"""BM25 lexical retriever, implemented from scratch.

BM25 (Okapi BM25) is the strong, standard sparse-retrieval baseline. It scores a chunk for
a query by summing, over the query terms, an inverse-document-frequency weight times a
saturating term-frequency component that is normalized by chunk length. Two parameters
tune it: ``k1`` (term-frequency saturation) and ``b`` (length normalization).

Implementing it directly - rather than importing a library - makes the ranking function
transparent and is the point of the exercise.

The index is inverted: each term maps to the chunks that contain it, with each chunk's BM25
term weight computed once at index time (the weight does not depend on the query). A query
adds the weights of its terms into one score array (see :mod:`hybrid_rag.core.postings`), in
query order, which gives each chunk the same sum as scoring it directly, so the ranking is
unchanged while only chunks sharing a term with the query are touched.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from hybrid_rag.core.postings import Postings, PostingsBuilder, top_k
from hybrid_rag.core.tokenization import tokenize
from hybrid_rag.core.types import Chunk, ScoredChunk


class BM25Retriever:
    """A from-scratch Okapi BM25 retriever over an inverted index.

    ``stem`` (default on) reduces query and chunk words to Porter stems, so different
    forms of a word match.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75, stem: bool = True) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self._k1 = k1
        self._b = b
        self._stem = stem
        self._chunks: list[Chunk] = []
        self._postings: Postings = Postings.empty()
        self._avg_len = 0.0

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Build the inverted index over ``chunks``."""
        self._chunks = list(chunks)
        builder = PostingsBuilder()
        lengths: list[int] = []
        for doc, chunk in enumerate(self._chunks):
            tokens = tokenize(chunk.text, stem=self._stem)
            lengths.append(len(tokens))
            builder.add_chunk(doc, Counter(tokens).items())
        postings = builder.build()
        self._avg_len = sum(lengths) / len(lengths) if lengths else 0.0
        # The length-dependent part of the BM25 denominator, fixed per chunk.
        length_norm = np.array(
            [self._k1 * (1 - self._b + self._b * n / self._avg_len) for n in lengths],
            dtype=np.float64,
        )
        idf = np.array(
            [self._compute_idf(int(df)) for df in postings.document_frequencies()],
            dtype=np.float64,
        )
        tf = postings.values
        # idf * tf * (k1 + 1) / (tf + length_norm): each chunk's score for one query term.
        postings.values = (
            idf[postings.term_of_entry] * tf * (self._k1 + 1) / (tf + length_norm[postings.ids])
        )
        self._postings = postings

    def retrieve(self, query: str, *, k: int = 10) -> list[ScoredChunk]:
        """Return the top-``k`` chunks by BM25 score."""
        if not self._chunks:
            return []
        scores = np.zeros(len(self._chunks), dtype=np.float64)
        for term in tokenize(query, stem=self._stem):
            posting = self._postings.get(term)
            if posting is not None:
                ids, weights = posting
                scores[ids] += weights
        return [
            ScoredChunk(chunk=self._chunks[doc], score=score) for doc, score in top_k(scores, k)
        ]

    def _compute_idf(self, df: int) -> float:
        # Okapi BM25 idf with the standard +0.5 smoothing; clamped at zero so terms in
        # more than half the corpus never contribute a negative score.
        n = len(self._chunks)
        return max(0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0))
