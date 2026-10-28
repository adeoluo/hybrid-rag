"""Tests for the RAG context builder and chunk persistence."""

from __future__ import annotations

from pathlib import Path

from hybrid_rag.core import Chunk
from hybrid_rag.persistence import ChunkStore
from hybrid_rag.rag import RagContext, build_context, extract_snippet
from hybrid_rag.retrievers import BM25Retriever


def _corpus() -> list[Chunk]:
    texts = {
        "c0": "Photosynthesis converts sunlight into glucose and oxygen in green plants.",
        "c1": "Chlorophyll is the pigment that absorbs light energy during photosynthesis.",
        "c2": "Dogs are loyal animals that enjoy long walks in the park.",
    }
    return [
        Chunk(id=cid, document_id=cid.replace("c", "doc"), text=text) for cid, text in texts.items()
    ]


# --------------------------------------------------------------------------- #
# Snippet extraction
# --------------------------------------------------------------------------- #
def test_extract_snippet_finds_relevant_window() -> None:
    text = " ".join(
        ["filler"] * 40 + ["the", "capital", "of", "France", "is", "Paris"] + ["filler"] * 40
    )
    snippet = extract_snippet(text, "capital France Paris", window=10)
    assert "Paris" in snippet


def test_extract_snippet_short_text_returned_whole() -> None:
    assert extract_snippet("short text", "anything", window=30) == "short text"


# --------------------------------------------------------------------------- #
# Context building
# --------------------------------------------------------------------------- #
def test_build_context_has_citations() -> None:
    retriever = BM25Retriever()
    retriever.index(_corpus())
    context = build_context("photosynthesis sunlight", retriever, k=2)
    assert isinstance(context, RagContext)
    assert context.num_sources >= 1
    assert "[1]" in context.context
    assert context.citations[0].marker == "[1]"


def test_build_context_respects_char_budget() -> None:
    retriever = BM25Retriever()
    retriever.index(_corpus())
    context = build_context("photosynthesis", retriever, k=3, max_chars=50)
    # The budget caps how many sources are included (at least one is always kept).
    assert len(context.context) <= 200
    assert context.num_sources >= 1


def test_build_context_citations_reference_chunks() -> None:
    retriever = BM25Retriever()
    retriever.index(_corpus())
    context = build_context("chlorophyll light", retriever, k=1)
    citation = context.citations[0]
    assert citation.chunk_id.startswith("c")
    assert citation.document_id.startswith("doc")


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_chunk_store_round_trip(tmp_path: Path) -> None:
    chunks = _corpus()
    path = tmp_path / "chunks.json"
    ChunkStore.save(chunks, path)
    loaded = ChunkStore.load(path)
    assert len(loaded) == len(chunks)
    assert loaded[0].id == chunks[0].id
    assert loaded[0].text == chunks[0].text


def test_loaded_chunks_are_indexable(tmp_path: Path) -> None:
    path = tmp_path / "chunks.json"
    ChunkStore.save(_corpus(), path)
    retriever = BM25Retriever()
    retriever.index(ChunkStore.load(path))
    results = retriever.retrieve("photosynthesis", k=1)
    assert results[0].chunk.id == "c0"
