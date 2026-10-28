"""Tests for the char-ngram and phrase retrievers and MMR reranking."""

from __future__ import annotations

import pytest

from hybrid_rag.core import Chunk
from hybrid_rag.mmr import MMRReranker
from hybrid_rag.retrievers import BM25Retriever, CharNgramRetriever, PhraseRetriever
from hybrid_rag.retrievers.char_ngram import char_ngrams


def _corpus() -> list[Chunk]:
    texts = {
        "c0": "Photosynthesis converts sunlight into glucose in plants.",
        "c1": "Reciprocal rank fusion merges ranked result lists.",
        "c2": "Dogs are loyal animals that enjoy long walks.",
        "c3": "The neural network learned representations from data.",
    }
    return [Chunk(id=cid, document_id="d", text=text) for cid, text in texts.items()]


# --------------------------------------------------------------------------- #
# Char n-gram
# --------------------------------------------------------------------------- #
def test_char_ngrams_basic() -> None:
    grams = char_ngrams("cat", 2)
    assert grams != (" c",)  # sanity: keys are strings not tuples
    assert "ca" in grams
    assert "at" in grams


def test_char_ngram_retrieves_despite_typo() -> None:
    retriever = CharNgramRetriever(n=3)
    retriever.index(_corpus())
    # Misspelled query still retrieves the photosynthesis chunk.
    results = retriever.retrieve("photosynthsis sunlght", k=1)
    assert results
    assert results[0].chunk.id == "c0"


def test_char_ngram_morphology_match() -> None:
    retriever = CharNgramRetriever(n=3)
    retriever.index(_corpus())
    # "neurons" shares trigrams with "neural".
    results = retriever.retrieve("neurons", k=1)
    assert results[0].chunk.id == "c3"


def test_char_ngram_empty_index() -> None:
    assert CharNgramRetriever().retrieve("x") == []


def test_char_ngram_validates() -> None:
    with pytest.raises(ValueError, match="n must"):
        CharNgramRetriever(n=0)


# --------------------------------------------------------------------------- #
# Phrase
# --------------------------------------------------------------------------- #
def test_phrase_retriever_rewards_exact_phrase() -> None:
    retriever = PhraseRetriever(max_n=3)
    retriever.index(_corpus())
    results = retriever.retrieve("reciprocal rank fusion", k=1)
    assert results[0].chunk.id == "c1"


def test_phrase_retriever_no_match() -> None:
    retriever = PhraseRetriever()
    retriever.index(_corpus())
    assert retriever.retrieve("quantum chromodynamics") == []


def test_phrase_retriever_validates() -> None:
    with pytest.raises(ValueError, match="max_n"):
        PhraseRetriever(max_n=0)


# --------------------------------------------------------------------------- #
# MMR
# --------------------------------------------------------------------------- #
def test_mmr_promotes_diversity() -> None:
    retriever = BM25Retriever()
    # Two near-duplicate chunks plus a distinct one.
    chunks = [
        Chunk(id="a", document_id="d", text="python programming language data science"),
        Chunk(id="b", document_id="d", text="python programming language data science tools"),
        Chunk(id="c", document_id="d", text="python machine learning neural networks"),
    ]
    retriever.index(chunks)
    candidates = retriever.retrieve("python", k=3)
    # With high diversity, the second pick should be the distinct chunk, not the duplicate.
    reranked = MMRReranker(diversity=0.8).rerank("python", candidates, k=2)
    assert len(reranked) == 2
    assert reranked[1].chunk.id == "c"


def test_mmr_pure_relevance_preserves_top() -> None:
    retriever = BM25Retriever()
    retriever.index(_corpus())
    candidates = retriever.retrieve("plants sunlight", k=3)
    reranked = MMRReranker(diversity=0.0).rerank("plants sunlight", candidates, k=3)
    assert reranked[0].id == candidates[0].id


def test_mmr_validates_diversity() -> None:
    with pytest.raises(ValueError, match="diversity"):
        MMRReranker(diversity=1.5)


def test_mmr_empty() -> None:
    assert MMRReranker().rerank("q", [], k=5) == []
