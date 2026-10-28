"""Retrieval-augmented generation: assembling cited context for a reader.

Retrieval returns chunks; a RAG system must turn them into a context block a language
model can answer from, with citations back to the sources. This module does the retrieval
half of RAG end to end:

- :func:`build_context` - retrieve, then format the top chunks into a numbered context
  block with source markers, respecting a character budget.
- :func:`extract_snippet` - pull the most query-relevant window from a chunk, for
  highlighting.
- :class:`RagContext` - the assembled context plus the ordered citations it was built from.

Generation itself (calling a model) is intentionally left to the caller; this project is
about the retrieval and grounding, not the generator.
"""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_rag.core.retriever import Retriever
from hybrid_rag.core.tokenization import tokenize
from hybrid_rag.core.types import ScoredChunk


@dataclass(frozen=True, slots=True)
class Citation:
    """A source used in an assembled context."""

    marker: str
    chunk_id: str
    document_id: str
    score: float


@dataclass(frozen=True, slots=True)
class RagContext:
    """An assembled context block with its ordered citations."""

    query: str
    context: str
    citations: tuple[Citation, ...]

    @property
    def num_sources(self) -> int:
        return len(self.citations)


def extract_snippet(text: str, query: str, *, window: int = 30) -> str:
    """Return the ``window``-word span of ``text`` most relevant to ``query``.

    Relevance of a window is the count of query terms it contains; ties resolve to the
    earliest window. Falls back to the leading window when no term matches.
    """
    words = text.split()
    if len(words) <= window:
        return text
    query_terms = set(tokenize(query))
    best_start = 0
    best_hits = -1
    for start in range(0, len(words) - window + 1):
        window_terms = set(tokenize(" ".join(words[start : start + window])))
        hits = len(query_terms & window_terms)
        if hits > best_hits:
            best_hits = hits
            best_start = start
    snippet = " ".join(words[best_start : best_start + window])
    prefix = "..." if best_start > 0 else ""
    suffix = "..." if best_start + window < len(words) else ""
    return f"{prefix}{snippet}{suffix}"


def build_context(
    query: str,
    retriever: Retriever,
    *,
    k: int = 5,
    max_chars: int = 2000,
    snippet_window: int = 60,
) -> RagContext:
    """Retrieve for ``query`` and assemble a cited, budget-bounded context block.

    Args:
        query: The user question.
        retriever: An indexed retriever.
        k: Maximum number of sources to consider.
        max_chars: Character budget for the assembled context.
        snippet_window: Words per source snippet.
    """
    results = retriever.retrieve(query, k=k)
    blocks: list[str] = []
    citations: list[Citation] = []
    used = 0
    for index, scored in enumerate(results, start=1):
        marker = f"[{index}]"
        snippet = extract_snippet(scored.chunk.text, query, window=snippet_window)
        block = f"{marker} {snippet}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
        citations.append(
            _citation(marker, scored),
        )
    return RagContext(query=query, context="\n\n".join(blocks), citations=tuple(citations))


def _citation(marker: str, scored: ScoredChunk) -> Citation:
    return Citation(
        marker=marker,
        chunk_id=scored.chunk.id,
        document_id=scored.chunk.document_id,
        score=scored.score,
    )
