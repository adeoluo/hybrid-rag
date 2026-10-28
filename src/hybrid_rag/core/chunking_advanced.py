"""Additional chunking strategies.

Beyond fixed-window and sentence chunking, two more strategies handle common document
shapes:

- :class:`RecursiveChunker` - splits on a priority list of separators (paragraphs, then
  sentences, then words), packing pieces up to a size budget. This keeps the largest
  natural boundaries intact and degrades gracefully, the approach popularized by modern
  RAG stacks.
- :class:`ParagraphChunker` - one chunk per blank-line-delimited paragraph, for already
  well-structured text.
"""

from __future__ import annotations

import re

from hybrid_rag.core.types import Chunk, Document

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\S+")


def _make_chunk(document: Document, index: int, text: str) -> Chunk:
    return Chunk(
        id=f"{document.id}#{index}",
        document_id=document.id,
        text=text,
        metadata=dict(document.metadata),
    )


class RecursiveChunker:
    """Splits text on a priority list of separators, packing up to a word budget."""

    def __init__(self, *, max_words: int = 120) -> None:
        if max_words < 1:
            raise ValueError("max_words must be at least 1")
        self._max_words = max_words

    def chunk(self, document: Document) -> list[Chunk]:
        pieces = self._split(document.text)
        chunks: list[Chunk] = []
        current: list[str] = []
        current_words = 0
        index = 0
        for piece in pieces:
            words = len(_WORD.findall(piece))
            if current and current_words + words > self._max_words:
                chunks.append(_make_chunk(document, index, " ".join(current)))
                index += 1
                current = []
                current_words = 0
            current.append(piece)
            current_words += words
        if current:
            chunks.append(_make_chunk(document, index, " ".join(current)))
        return chunks

    def _split(self, text: str) -> list[str]:
        # Try paragraphs first; split any that are still over budget into sentences, then
        # fall back to word windows for pathologically long sentences.
        pieces: list[str] = []
        for paragraph in _PARAGRAPH.split(text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(_WORD.findall(paragraph)) <= self._max_words:
                pieces.append(paragraph)
                continue
            for sentence in _SENTENCE.split(paragraph):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(_WORD.findall(sentence)) <= self._max_words:
                    pieces.append(sentence)
                else:
                    pieces.extend(self._word_windows(sentence))
        return pieces

    def _word_windows(self, text: str) -> list[str]:
        words = _WORD.findall(text)
        return [
            " ".join(words[i : i + self._max_words]) for i in range(0, len(words), self._max_words)
        ]


class ParagraphChunker:
    """One chunk per blank-line-delimited paragraph."""

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for paragraph in _PARAGRAPH.split(document.text):
            paragraph = paragraph.strip()
            if paragraph:
                chunks.append(_make_chunk(document, index, paragraph))
                index += 1
        return chunks
