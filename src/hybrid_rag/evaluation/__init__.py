"""Retrieval evaluation: metrics and a harness."""

from __future__ import annotations

from hybrid_rag.evaluation.comparison import (
    ComparisonResult,
    compare_retrievers,
    hit_rate,
    r_precision,
    success_at_k,
)
from hybrid_rag.evaluation.harness import (
    LabeledQuery,
    RetrievalReport,
    evaluate_retriever,
)
from hybrid_rag.evaluation.metrics import (
    average_precision,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "ComparisonResult",
    "LabeledQuery",
    "RetrievalReport",
    "average_precision",
    "compare_retrievers",
    "dcg_at_k",
    "evaluate_retriever",
    "hit_rate",
    "ndcg_at_k",
    "precision_at_k",
    "r_precision",
    "recall_at_k",
    "reciprocal_rank",
    "success_at_k",
]
