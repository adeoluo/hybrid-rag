"""A from-scratch neural dense retriever.

Unlike the sparse lexical retrievers (BM25, TF-IDF), this retriever learns a dense
embedding for every word and represents a chunk or query as the mean of its word
vectors, ranking by cosine similarity. The embeddings are trained from scratch in
pure Python with a skip-gram objective and negative sampling over the indexed
corpus, so semantically related words move together in the vector space even when
they are not the same token.

The honest caveat, documented in the README, is that learning useful dense
semantics needs far more data than a small bundled corpus provides; on this scale
a well-tuned lexical retriever is a very strong baseline. The value here is the
from-scratch implementation of the training and the dense representation, measured
honestly against the lexical methods rather than assumed to beat them.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from hybrid_rag.core.tokenization import tokenize
from hybrid_rag.core.types import Chunk, ScoredChunk

Vector = list[float]


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class DenseRetriever:
    """A dense retriever over learned word embeddings, ranked by cosine similarity."""

    def __init__(
        self,
        *,
        dim: int = 48,
        epochs: int = 40,
        window: int = 3,
        negatives: int = 5,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 0,
    ) -> None:
        self.dim = dim
        self.epochs = epochs
        self.window = window
        self.negatives = negatives
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed
        self._chunks: list[Chunk] = []
        self._vectors: list[Vector] = []
        self._norms: list[float] = []
        self._in: dict[str, Vector] = {}
        self._out: dict[str, Vector] = {}
        self._vocab: list[str] = []
        self._sample_table: list[str] = []

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Train word embeddings on ``chunks`` and embed each chunk."""
        self._chunks = list(chunks)
        tokenized = [tokenize(chunk.text) for chunk in self._chunks]
        self._train(tokenized)
        self._vectors = [self._embed(tokens) for tokens in tokenized]
        self._norms = [_norm(vector) for vector in self._vectors]

    def retrieve(self, query: str, *, k: int = 10) -> list[ScoredChunk]:
        """Return the top-``k`` chunks by cosine similarity to the query embedding."""
        if not self._chunks:
            return []
        query_vector = self._embed(tokenize(query))
        query_norm = _norm(query_vector)
        if query_norm == 0.0:
            return []
        scored: list[ScoredChunk] = []
        for index, vector in enumerate(self._vectors):
            denominator = self._norms[index] * query_norm
            similarity = _dot(vector, query_vector) / denominator if denominator else 0.0
            scored.append(ScoredChunk(chunk=self._chunks[index], score=similarity))
        scored.sort(key=lambda item: (-item.score, item.chunk.id))
        return scored[:k]

    def embedding(self, token: str) -> Vector | None:
        """Return the learned input embedding for ``token`` if it is in the vocabulary."""
        vector = self._in.get(token)
        return list(vector) if vector is not None else None

    def _train(self, tokenized: list[list[str]]) -> None:
        rng = random.Random(self.seed)
        counts: dict[str, int] = {}
        for tokens in tokenized:
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
        self._vocab = sorted(word for word, count in counts.items() if count >= self.min_count)
        scale = 0.5 / self.dim
        self._in = {
            word: [rng.uniform(-scale, scale) for _ in range(self.dim)] for word in self._vocab
        }
        self._out = {word: [0.0] * self.dim for word in self._vocab}
        known = set(self._vocab)

        # A unigram sampling table (slightly flattened) for negative sampling.
        self._sample_table = []
        for word in self._vocab:
            weight = max(1, int(counts[word] ** 0.75))
            self._sample_table.extend([word] * weight)

        corpus = [[token for token in tokens if token in known] for tokens in tokenized]
        for _ in range(self.epochs):
            for tokens in corpus:
                for center_index, center in enumerate(tokens):
                    lo = max(0, center_index - self.window)
                    hi = min(len(tokens), center_index + self.window + 1)
                    for context_index in range(lo, hi):
                        if context_index == center_index:
                            continue
                        self._step(center, tokens[context_index], rng)

    def _step(self, center: str, context: str, rng: random.Random) -> None:
        v_in = self._in[center]
        samples: list[tuple[str, float]] = [(context, 1.0)]
        for _ in range(self.negatives):
            negative = rng.choice(self._sample_table)
            if negative != context:
                samples.append((negative, 0.0))
        gradient_in = [0.0] * self.dim
        for word, label in samples:
            v_out = self._out[word]
            score = _dot(v_in, v_out)
            error = _sigmoid(score) - label
            step = self.learning_rate * error
            for i in range(self.dim):
                gradient_in[i] += step * v_out[i]
                v_out[i] -= step * v_in[i]
        for i in range(self.dim):
            v_in[i] -= gradient_in[i]

    def _embed(self, tokens: list[str]) -> Vector:
        vectors = [self._in[token] for token in tokens if token in self._in]
        if not vectors:
            return [0.0] * self.dim
        total = [0.0] * self.dim
        for vector in vectors:
            for i in range(self.dim):
                total[i] += vector[i]
        count = len(vectors)
        return [value / count for value in total]


def _dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _norm(vector: Vector) -> float:
    return math.sqrt(_dot(vector, vector))
