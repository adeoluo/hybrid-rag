"""Text tokenization shared across retrievers.

A single, explicit tokenizer keeps BM25 and the dense retriever consistent: they index and
query the same token stream. Tokenization lowercases, splits on non-alphanumerics, and
optionally removes a small English stopword list and reduces words to their Porter stems -
decisions that materially affect retrieval quality, so they are centralized and documented
here.
"""

from __future__ import annotations

import re

from hybrid_rag.core.stemming import stem_all

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "as",
        "by",
        "with",
        "from",
        "into",
        "about",
        "which",
        "who",
        "whom",
        "what",
        "when",
        "where",
        "how",
        "why",
    }
)


def tokenize(text: str, *, remove_stopwords: bool = True, stem: bool = False) -> list[str]:
    """Lowercase, split, and optionally drop stopwords from and stem ``text``."""
    tokens = _TOKEN.findall(text.lower())
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    if stem:
        tokens = stem_all(tokens)
    return tokens
