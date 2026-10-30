"""A large-workload scale benchmark for the hybrid retrieval pipeline.

Where ``benchmark.py`` times each pipeline on one corpus, this script asks how the
flagship ``hybrid`` pipeline (BM25 + TF-IDF fusion) scales: it indexes
progressively larger synthetic chunk corpora and runs a fixed labeled query set
against each, reporting wall-clock time, query throughput, peak memory (via the
stdlib ``tracemalloc``), and recall@5 so cost and quality are visible together.

Each query has one planted relevant chunk that repeats the query terms; the rest
of the corpus is noise drawn from the same vocabulary, so the true positive must
be distinguished by term weighting rather than a unique token, and recall decays
as the field of distractors grows.

The data is synthetic and every retriever here is from-scratch pure Python
(including a classical sparse TF-IDF, which is lexical by design, not neural): it
is not meant to be competitive on wall-clock with compiled indexes or vector
databases. The point is to show how this implementation scales, honestly.

Run with:

    uv run python benchmarks/scale.py
"""

from __future__ import annotations

import random
import time
import tracemalloc
from dataclasses import dataclass

from hybrid_rag.core import Chunk
from hybrid_rag.evaluation.harness import LabeledQuery, evaluate_retriever
from hybrid_rag.pipelines import build_pipeline

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
    "revolution",
    "mitochondria",
    "evolution",
    "cell",
    "energy",
    "sunlight",
    "ocean",
    "mountain",
    "pigment",
    "spacetime",
    "glucose",
    "oxygen",
]

_CORPUS_SIZES: tuple[int, ...] = (500, 1000, 2000, 4000)
_NUM_QUERIES = 80
_K = 5
_PIPELINE = "hybrid"


@dataclass(frozen=True, slots=True)
class Workload:
    """A synthetic chunk corpus plus the labeled queries that probe it."""

    chunks: list[Chunk]
    queries: list[LabeledQuery]


def build_workload(num_chunks: int, num_queries: int, *, seed: int = 0) -> Workload:
    """Build a corpus with one planted relevant chunk per query."""
    rng = random.Random(seed)
    chunks: list[Chunk] = []
    queries: list[LabeledQuery] = []

    for query_index in range(num_queries):
        terms = rng.sample(_WORDS, k=3)
        relevant_id = f"rel{query_index}"
        queries.append(LabeledQuery(query=" ".join(terms), relevant={relevant_id}))
        filler = [rng.choice(_WORDS) for _ in range(34)]
        words = terms * 2 + filler
        rng.shuffle(words)
        chunks.append(Chunk(id=relevant_id, document_id=f"d{query_index}", text=" ".join(words)))

    for distractor_index in range(num_chunks - num_queries):
        words = [rng.choice(_WORDS) for _ in range(40)]
        chunks.append(
            Chunk(
                id=f"noise{distractor_index}",
                document_id=f"dn{distractor_index}",
                text=" ".join(words),
            )
        )

    rng.shuffle(chunks)
    return Workload(chunks=chunks, queries=queries)


@dataclass(frozen=True, slots=True)
class ScaleResult:
    """One row of the scale table."""

    chunks: int
    seconds: float
    queries_per_sec: float
    peak_mib: float
    recall: float


def measure(num_chunks: int, *, k: int = _K) -> ScaleResult:
    """Index ``num_chunks`` chunks with the hybrid pipeline and evaluate recall@k.

    Timing and throughput are measured without ``tracemalloc`` (whose per-allocation
    hook would dominate this allocation-heavy pipeline and understate throughput).
    Peak memory is then captured in a separate ``tracemalloc`` pass over the same
    workload, so each number reflects what it claims to.
    """
    workload = build_workload(num_chunks, _NUM_QUERIES)

    retriever = build_pipeline(_PIPELINE)
    index_start = time.perf_counter()
    retriever.index(workload.chunks)
    index_elapsed = time.perf_counter() - index_start

    query_start = time.perf_counter()
    report = evaluate_retriever(retriever, workload.queries, k=k)
    query_elapsed = time.perf_counter() - query_start

    peak = _peak_memory(workload, k=k)

    return ScaleResult(
        chunks=num_chunks,
        seconds=index_elapsed + query_elapsed,
        queries_per_sec=len(workload.queries) / query_elapsed if query_elapsed else 0.0,
        peak_mib=peak / (1024 * 1024),
        recall=report.recall,
    )


def _peak_memory(workload: Workload, *, k: int) -> int:
    """Return the peak bytes to index the corpus and serve a few queries."""
    tracemalloc.start()
    retriever = build_pipeline(_PIPELINE)
    retriever.index(workload.chunks)
    for labeled in workload.queries[:5]:
        retriever.retrieve(labeled.query, k=k)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def main() -> None:
    print("Synthetic corpus; from-scratch pure-Python hybrid retrieval (not a vector DB).")
    print(
        f"Flagship '{_PIPELINE}' pipeline, {_NUM_QUERIES} queries, recall@{_K} vs planted chunk.\n"
    )
    header = f"{'chunks':>8}{'seconds':>10}{'queries/sec':>14}{'peak MiB':>11}{'recall@5':>11}"
    print(header)
    print("-" * len(header))
    for num_chunks in _CORPUS_SIZES:
        result = measure(num_chunks)
        print(
            f"{result.chunks:>8}{result.seconds:>10.3f}{result.queries_per_sec:>14.0f}"
            f"{result.peak_mib:>11.1f}{result.recall:>11.2f}"
        )


if __name__ == "__main__":
    main()
