"""Core data types for retrieval.

Retrieval operates over three small, immutable records:

- :class:`Document` - a source text with an id and optional metadata.
- :class:`Chunk` - a retrievable span of a document (documents are split for retrieval).
- :class:`ScoredChunk` - a chunk paired with a relevance score from a retriever.

Keeping these minimal and immutable lets retrievers, fusion, and rerankers compose without
coupling to any particular index implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """A source document to be indexed and retrieved over."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Document.id must be non-empty")
        if not self.text:
            raise ValueError("Document.text must be non-empty")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable span of a document."""

    id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Chunk.id must be non-empty")
        if not self.text:
            raise ValueError("Chunk.text must be non-empty")


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk with a relevance score assigned by a retriever or reranker."""

    chunk: Chunk
    score: float

    @property
    def id(self) -> str:
        return self.chunk.id
