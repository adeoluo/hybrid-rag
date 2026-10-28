"""Tests for query processing and advanced chunking."""

from __future__ import annotations

import pytest

from hybrid_rag.core import Chunk, Document
from hybrid_rag.core.chunking_advanced import ParagraphChunker, RecursiveChunker
from hybrid_rag.query import SynonymExpander, expand_with_prf
from hybrid_rag.retrievers import BM25Retriever


# --------------------------------------------------------------------------- #
# Synonym expansion
# --------------------------------------------------------------------------- #
def test_synonym_expander_appends_synonyms() -> None:
    expanded = SynonymExpander().expand("I need a fast car")
    assert "automobile" in expanded
    assert "quick" in expanded or "rapid" in expanded


def test_synonym_expander_no_match_unchanged() -> None:
    assert SynonymExpander().expand("xyzzy plugh") == "xyzzy plugh"


def test_synonym_expander_custom_map() -> None:
    expander = SynonymExpander({"cat": ("feline",)})
    assert "feline" in expander.expand("the cat")


# --------------------------------------------------------------------------- #
# Pseudo-relevance feedback
# --------------------------------------------------------------------------- #
def _corpus() -> list[Chunk]:
    texts = {
        "c0": "Photosynthesis lets plants convert sunlight into glucose and oxygen.",
        "c1": "Chlorophyll is the green pigment that absorbs light energy in plants.",
        "c2": "Dogs are loyal animals that enjoy walks in the park.",
    }
    return [Chunk(id=cid, document_id="d", text=text) for cid, text in texts.items()]


def test_prf_adds_terms() -> None:
    retriever = BM25Retriever()
    retriever.index(_corpus())
    expanded = expand_with_prf("plants sunlight", retriever, expansion_terms=2)
    assert len(expanded.split()) > len(["plants", "sunlight"])


def test_prf_empty_results_unchanged() -> None:
    retriever = BM25Retriever()
    retriever.index(_corpus())
    assert expand_with_prf("xylophone", retriever) == "xylophone"


def test_prf_validates_terms() -> None:
    retriever = BM25Retriever()
    retriever.index(_corpus())
    with pytest.raises(ValueError, match="expansion_terms"):
        expand_with_prf("plants", retriever, expansion_terms=0)


# --------------------------------------------------------------------------- #
# Recursive chunker
# --------------------------------------------------------------------------- #
def test_recursive_chunker_respects_paragraphs() -> None:
    text = "First paragraph here.\n\nSecond paragraph follows with more words."
    chunks = RecursiveChunker(max_words=20).chunk(Document(id="d", text=text))
    assert len(chunks) >= 1
    assert all(c.document_id == "d" for c in chunks)


def test_recursive_chunker_splits_long_text() -> None:
    text = " ".join(f"word{i}" for i in range(100))
    chunks = RecursiveChunker(max_words=25).chunk(Document(id="d", text=text))
    assert len(chunks) >= 3


def test_recursive_chunker_validates() -> None:
    with pytest.raises(ValueError, match="max_words"):
        RecursiveChunker(max_words=0)


# --------------------------------------------------------------------------- #
# Paragraph chunker
# --------------------------------------------------------------------------- #
def test_paragraph_chunker_one_per_paragraph() -> None:
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = ParagraphChunker().chunk(Document(id="d", text=text))
    assert len(chunks) == 3
    assert chunks[0].text == "Para one."


def test_paragraph_chunker_ignores_blank() -> None:
    chunks = ParagraphChunker().chunk(Document(id="d", text="only one\n\n\n\n"))
    assert len(chunks) == 1
