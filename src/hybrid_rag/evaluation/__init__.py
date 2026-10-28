"""Retrieval evaluation: metrics and a harness."""

from __future__ import annotations

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
    "LabeledQuery",
    "RetrievalReport",
    "average_precision",
    "dcg_at_k",
    "evaluate_retriever",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
