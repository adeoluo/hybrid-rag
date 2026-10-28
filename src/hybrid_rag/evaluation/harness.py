"""A retrieval evaluation harness.

Given a set of labeled queries - each a query string plus the ids of the chunks that
should be retrieved - :func:`evaluate_retriever` runs a retriever and aggregates the
standard metrics across queries. This is what turns "we built a retriever" into "our
retriever achieves recall@10 of X", and it lets any two retrievers be compared on the same
queries under identical conditions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from hybrid_rag.core.retriever import Retriever
from hybrid_rag.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


@dataclass(frozen=True, slots=True)
class LabeledQuery:
    """A query with its relevant chunk ids and optional graded gains."""

    query: str
    relevant: set[str]
    gains: Mapping[str, float] = field(default_factory=dict)

    def effective_gains(self) -> dict[str, float]:
        """Return graded gains, defaulting to 1.0 for each relevant id."""
        if self.gains:
            return dict(self.gains)
        return dict.fromkeys(self.relevant, 1.0)


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    """Aggregated retrieval metrics across a query set."""

    k: int
    recall: float
    precision: float
    mrr: float
    ndcg: float
    num_queries: int

    def format(self) -> str:
        """Render a one-line summary."""
        return (
            f"recall@{self.k}={self.recall:.3f}  "
            f"precision@{self.k}={self.precision:.3f}  "
            f"MRR={self.mrr:.3f}  nDCG@{self.k}={self.ndcg:.3f}  "
            f"(N={self.num_queries})"
        )


def evaluate_retriever(
    retriever: Retriever,
    queries: Sequence[LabeledQuery],
    *,
    k: int = 10,
) -> RetrievalReport:
    """Run ``retriever`` over ``queries`` and aggregate metrics at cutoff ``k``.

    The retriever must already be indexed. Metrics are averaged over queries.
    """
    if not queries:
        raise ValueError("cannot evaluate over an empty query set")

    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []

    for labeled in queries:
        retrieved = [scored.id for scored in retriever.retrieve(labeled.query, k=k)]
        recalls.append(recall_at_k(retrieved, labeled.relevant, k))
        precisions.append(precision_at_k(retrieved, labeled.relevant, k))
        reciprocal_ranks.append(reciprocal_rank(retrieved, labeled.relevant))
        ndcgs.append(ndcg_at_k(retrieved, labeled.effective_gains(), k))

    n = len(queries)
    return RetrievalReport(
        k=k,
        recall=sum(recalls) / n,
        precision=sum(precisions) / n,
        mrr=sum(reciprocal_ranks) / n,
        ndcg=sum(ndcgs) / n,
        num_queries=n,
    )
