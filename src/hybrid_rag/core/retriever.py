"""The retriever interface.

A :class:`Retriever` indexes a set of chunks and returns the top-k most relevant for a
query, as scored chunks. BM25, the dense retriever, the hybrid fusion, and the reranked
pipeline all satisfy this one interface, so they compose and are evaluated identically.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from hybrid_rag.core.types import Chunk, ScoredChunk


@runtime_checkable
class Retriever(Protocol):
    """Indexes chunks and retrieves the top-k for a query."""

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Build the retriever's index over ``chunks``."""
        ...

    def retrieve(self, query: str, *, k: int = 10) -> list[ScoredChunk]:
        """Return up to ``k`` scored chunks, highest score first."""
        ...
