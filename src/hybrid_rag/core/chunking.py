"""Chunking strategies: splitting documents into retrievable spans.

Chunking is one of the highest-leverage decisions in a retrieval system: chunks that are
too large dilute relevance, while chunks that are too small lose context. This module
provides two transparent strategies with a shared interface:

- :class:`FixedTokenChunker` - fixed-size windows with configurable overlap.
- :class:`SentenceChunker` - sentence-aware packing up to a token budget.

Both are deterministic, so an index is reproducible from its documents.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from hybrid_rag.core.types import Chunk, Document

_WORD = re.compile(r"\S+")
_SENTENCE = re.compile(r"[^.!?]+[.!?]?")


@runtime_checkable
class Chunker(Protocol):
    """Splits a document into one or more chunks."""

    def chunk(self, document: Document) -> list[Chunk]:
        """Return the chunks of ``document`` in order."""
        ...


class FixedTokenChunker:
    """Splits text into fixed-size word windows with overlap.

    Args:
        chunk_size: Maximum words per chunk.
        overlap: Number of words shared between consecutive chunks, for context continuity.
    """

    def __init__(self, *, chunk_size: int = 120, overlap: int = 20) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        words = _WORD.findall(document.text)
        if not words:
            return []
        step = self._chunk_size - self._overlap
        chunks: list[Chunk] = []
        index = 0
        start = 0
        while start < len(words):
            window = words[start : start + self._chunk_size]
            chunks.append(_make_chunk(document, index, " ".join(window)))
            index += 1
            if start + self._chunk_size >= len(words):
                break
            start += step
        return chunks


class SentenceChunker:
    """Packs whole sentences into chunks up to a word budget.

    Keeps sentence boundaries intact, which preserves local context better than fixed
    windows for prose.
    """

    def __init__(self, *, max_words: int = 120) -> None:
        if max_words < 1:
            raise ValueError("max_words must be at least 1")
        self._max_words = max_words

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = [s.strip() for s in _SENTENCE.findall(document.text) if s.strip()]
        chunks: list[Chunk] = []
        current: list[str] = []
        current_words = 0
        index = 0
        for sentence in sentences:
            words = len(_WORD.findall(sentence))
            if current and current_words + words > self._max_words:
                chunks.append(_make_chunk(document, index, " ".join(current)))
                index += 1
                current = []
                current_words = 0
            current.append(sentence)
            current_words += words
        if current:
            chunks.append(_make_chunk(document, index, " ".join(current)))
        return chunks


def chunk_documents(documents: Sequence[Document], chunker: Chunker) -> list[Chunk]:
    """Apply ``chunker`` to every document and return all chunks."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunker.chunk(document))
    return chunks


def _make_chunk(document: Document, index: int, text: str) -> Chunk:
    return Chunk(
        id=f"{document.id}#{index}",
        document_id=document.id,
        text=text,
        metadata=dict(document.metadata),
    )
