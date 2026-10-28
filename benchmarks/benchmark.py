"""A micro-benchmark for the retrieval pipelines.

Measures index-build and query throughput for each pipeline over a synthetic corpus, so the
README can quote concrete numbers. Deterministic and offline; run with:

    uv run python benchmarks/benchmark.py
"""

from __future__ import annotations

import random
import time

from hybrid_rag.core import Chunk
from hybrid_rag.pipelines import available_pipelines, build_pipeline

_WORDS = [
    "neural", "network", "photosynthesis", "gravity", "retrieval", "fusion", "ranking",
    "embedding", "vector", "quantum", "revolution", "mitochondria", "evolution", "cell",
    "energy", "sunlight", "ocean", "mountain", "pigment", "spacetime", "glucose", "oxygen",
]


def _synthetic_corpus(num_chunks: int, *, seed: int = 0) -> list[Chunk]:
    rng = random.Random(seed)
    chunks: list[Chunk] = []
    for i in range(num_chunks):
        words = [rng.choice(_WORDS) for _ in range(40)]
        chunks.append(Chunk(id=f"c{i}", document_id=f"d{i}", text=" ".join(words)))
    return chunks


def main() -> None:
    corpus = _synthetic_corpus(500)
    queries = [" ".join(random.Random(q).choices(_WORDS, k=4)) for q in range(200)]

    print(f"Corpus: {len(corpus)} chunks | Queries: {len(queries)}\n")
    print(f"{'pipeline':<12} {'index (ms)':>12} {'queries/sec':>14}")
    print("-" * 40)
    for name in available_pipelines():
        retriever = build_pipeline(name)

        start = time.perf_counter()
        retriever.index(corpus)
        index_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        for query in queries:
            retriever.retrieve(query, k=10)
        elapsed = time.perf_counter() - start
        qps = len(queries) / elapsed if elapsed > 0 else float("inf")

        print(f"{name:<12} {index_ms:>12.1f} {qps:>14.0f}")


if __name__ == "__main__":
    main()
