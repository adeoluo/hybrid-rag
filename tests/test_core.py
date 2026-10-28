"""Tests for core types, tokenization, and chunking."""

from __future__ import annotations

import pytest

from hybrid_rag.core import (
    Chunk,
    Document,
    FixedTokenChunker,
    SentenceChunker,
    chunk_documents,
    tokenize,
)


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #
def test_document_validates() -> None:
    with pytest.raises(ValueError, match="id"):
        Document(id="", text="x")
    with pytest.raises(ValueError, match="text"):
        Document(id="d", text="")


def test_chunk_validates() -> None:
    with pytest.raises(ValueError, match="text"):
        Chunk(id="c", document_id="d", text="")


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #
def test_tokenize_removes_stopwords() -> None:
    assert tokenize("The quick brown fox") == ["quick", "brown", "fox"]


def test_tokenize_keeps_stopwords_when_asked() -> None:
    assert "the" in tokenize("the fox", remove_stopwords=False)


def test_tokenize_lowercases_and_splits_punctuation() -> None:
    assert tokenize("Hello, World!", remove_stopwords=False) == ["hello", "world"]


# --------------------------------------------------------------------------- #
# Fixed-token chunking
# --------------------------------------------------------------------------- #
def test_fixed_chunker_splits_with_overlap() -> None:
    text = " ".join(str(i) for i in range(100))
    chunker = FixedTokenChunker(chunk_size=30, overlap=10)
    chunks = chunker.chunk(Document(id="d", text=text))
    assert len(chunks) > 1
    assert all(c.document_id == "d" for c in chunks)
    # Overlap: last 10 words of chunk 0 begin chunk 1.
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-10:] == second_words[:10]


def test_fixed_chunker_single_chunk_for_short_text() -> None:
    chunker = FixedTokenChunker(chunk_size=100, overlap=0)
    chunks = chunker.chunk(Document(id="d", text="short text here"))
    assert len(chunks) == 1
    assert chunks[0].id == "d#0"


def test_fixed_chunker_validates() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        FixedTokenChunker(chunk_size=0)
    with pytest.raises(ValueError, match="overlap"):
        FixedTokenChunker(chunk_size=10, overlap=10)


# --------------------------------------------------------------------------- #
# Sentence chunking
# --------------------------------------------------------------------------- #
def test_sentence_chunker_packs_sentences() -> None:
    text = "First sentence here. Second one follows. Third is last."
    chunker = SentenceChunker(max_words=6)
    chunks = chunker.chunk(Document(id="d", text=text))
    assert len(chunks) >= 2
    # Each chunk should end at a sentence boundary.
    assert chunks[0].text.endswith(".")


def test_sentence_chunker_validates() -> None:
    with pytest.raises(ValueError, match="max_words"):
        SentenceChunker(max_words=0)


# --------------------------------------------------------------------------- #
# chunk_documents
# --------------------------------------------------------------------------- #
def test_chunk_documents_across_corpus() -> None:
    docs = [
        Document(id="a", text=" ".join(str(i) for i in range(50))),
        Document(id="b", text=" ".join(str(i) for i in range(50))),
    ]
    chunks = chunk_documents(docs, FixedTokenChunker(chunk_size=20, overlap=5))
    assert {c.document_id for c in chunks} == {"a", "b"}


def test_entrypoint(capsys: pytest.CaptureFixture[str]) -> None:
    from hybrid_rag.__main__ import main

    main()
    assert "hybrid-rag" in capsys.readouterr().out
