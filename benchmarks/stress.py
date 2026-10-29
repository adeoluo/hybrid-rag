"""A production-style stress benchmark for the retrieval service.

Indexes a synthetic corpus with every pipeline, then drives the full retrieve
handler under sustained, concurrent load and reports throughput with p50, p95,
and p99 latency per pipeline. Deterministic and offline; run with:

    uv run python benchmarks/stress.py
"""

from __future__ import annotations

import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from hybrid_rag.core.chunking import FixedTokenChunker, chunk_documents
from hybrid_rag.core.types import Document
from hybrid_rag.pipelines import available_pipelines, build_pipeline
from hybrid_rag.service import RetrievalService

_WORDS = [
    "neural",
    "network",
    "photosynthesis",
    "gravity",
    "retrieval",
    "fusion",
    "ranking",
    "embedding",
    "vector",
    "quantum",
    "evolution",
    "energy",
    "ocean",
    "mountain",
    "pigment",
    "glucose",
    "oxygen",
    "spacetime",
]


def _synthetic_service(num_documents: int, *, seed: int = 0) -> RetrievalService:
    rng = random.Random(seed)
    documents = [
        Document(id=f"d{index}", text=" ".join(rng.choice(_WORDS) for _ in range(120)))
        for index in range(num_documents)
    ]
    chunks = chunk_documents(documents, FixedTokenChunker(chunk_size=200, overlap=0))
    retrievers = {}
    for name in available_pipelines():
        retriever = build_pipeline(name)
        retriever.index(chunks)
        retrievers[name] = retriever
    return RetrievalService(retrievers=retrievers, chunk_count=len(chunks))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    service = _synthetic_service(300)
    rng = random.Random(0)
    queries = [" ".join(rng.choice(_WORDS) for _ in range(4)) for _ in range(500)]

    print(f"Chunks: {service.chunk_count} | Queries per pipeline: {len(queries)}\n")
    print(f"{'pipeline':<12}{'throughput':>14}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}")
    print("-" * 56)
    for pipeline in available_pipelines():

        def one(query: str, pipeline: str = pipeline) -> float:
            start = time.perf_counter()
            service.retrieve(query, pipeline, 5)
            return (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            latencies = list(pool.map(one, queries))
        wall = time.perf_counter() - start
        print(
            f"{pipeline:<12}{len(queries) / wall:>11.0f}/s"
            f"{statistics.median(latencies):>10.2f}"
            f"{_percentile(latencies, 0.95):>10.2f}"
            f"{_percentile(latencies, 0.99):>10.2f}"
        )


if __name__ == "__main__":
    main()
