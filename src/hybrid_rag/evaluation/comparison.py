"""Additional retrieval metrics and a bootstrap comparison of two retrievers.

Extends the metric set with:

- ``success_at_k`` / ``hit_rate`` - whether any relevant item is in the top k (a lenient,
  user-facing "did we find something" signal).
- ``r_precision`` - precision at rank R, where R is the number of relevant items, a
  cutoff-free summary of ranking quality.

And adds :func:`compare_retrievers`, which evaluates two retrievers on the same queries and
reports whether the difference in a chosen metric is statistically significant via a paired
bootstrap over per-query scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hybrid_rag.core.retriever import Retriever
from hybrid_rag.evaluation.harness import LabeledQuery
from hybrid_rag.evaluation.metrics import recall_at_k, reciprocal_rank


def success_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant item appears in the top ``k``, else 0.0."""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def hit_rate(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1.0 if any relevant item was retrieved at all, else 0.0."""
    return 1.0 if set(retrieved) & relevant else 0.0


def r_precision(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Precision at rank R, where R is the number of relevant items."""
    if not relevant:
        return 0.0
    r = len(relevant)
    top = retrieved[:r]
    return sum(1 for item in top if item in relevant) / r


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """The outcome of comparing two retrievers on a metric."""

    metric: str
    mean_a: float
    mean_b: float
    p_value: float
    significant: bool


def _per_query_metric(
    retriever: Retriever,
    queries: Sequence[LabeledQuery],
    metric: str,
    k: int,
) -> list[float]:
    scores: list[float] = []
    for query in queries:
        retrieved = [scored.id for scored in retriever.retrieve(query.query, k=k)]
        if metric == "recall":
            scores.append(recall_at_k(retrieved, query.relevant, k))
        elif metric == "mrr":
            scores.append(reciprocal_rank(retrieved, query.relevant))
        elif metric == "success":
            scores.append(success_at_k(retrieved, query.relevant, k))
        else:
            raise ValueError(f"unknown metric {metric!r}")
    return scores


def compare_retrievers(
    retriever_a: Retriever,
    retriever_b: Retriever,
    queries: Sequence[LabeledQuery],
    *,
    metric: str = "mrr",
    k: int = 10,
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> ComparisonResult:
    """Compare two indexed retrievers on a metric with a paired bootstrap test.

    Both retrievers must already be indexed on the same chunks.
    """
    if not queries:
        raise ValueError("cannot compare over an empty query set")
    scores_a = _per_query_metric(retriever_a, queries, metric, k)
    scores_b = _per_query_metric(retriever_b, queries, metric, k)
    differences = [a - b for a, b in zip(scores_a, scores_b, strict=True)]
    observed = sum(differences) / len(differences)

    import random

    rng = random.Random(seed)
    n = len(differences)
    count_opposite = 0
    for _ in range(resamples):
        resample_mean = sum(differences[rng.randrange(n)] for _ in range(n)) / n
        if (observed >= 0 and resample_mean <= 0) or (observed < 0 and resample_mean >= 0):
            count_opposite += 1
    p_value = min(1.0, 2.0 * count_opposite / resamples)

    return ComparisonResult(
        metric=metric,
        mean_a=sum(scores_a) / len(scores_a),
        mean_b=sum(scores_b) / len(scores_b),
        p_value=p_value,
        significant=p_value < alpha,
    )
