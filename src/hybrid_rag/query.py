"""Query processing: expansion and normalization.

Retrieval quality often hinges on the query as much as the index. A query that uses
different words than the documents ("car" versus "automobile") retrieves poorly. These
transforms improve recall before retrieval runs:

- :class:`SynonymExpander` - append known synonyms so lexical retrievers match more
  surface forms.
- :func:`expand_with_prf` - pseudo-relevance feedback: run an initial retrieval, harvest
  the most informative terms from the top results, and append them to the query.

Both return a new query string, so they compose and slot in front of any retriever.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from hybrid_rag.core.retriever import Retriever
from hybrid_rag.core.tokenization import tokenize

# A small, illustrative synonym map. In production this would be a curated thesaurus or a
# learned expansion model; here it is explicit and inspectable.
DEFAULT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "car": ("automobile", "vehicle"),
    "doctor": ("physician",),
    "big": ("large", "huge"),
    "buy": ("purchase",),
    "fast": ("quick", "rapid"),
    "energy": ("power",),
    "picture": ("image", "photo"),
    "movie": ("film",),
    "sea": ("ocean",),
    "smart": ("intelligent", "clever"),
}


class SynonymExpander:
    """Appends known synonyms of query terms to broaden lexical matching."""

    def __init__(self, synonyms: Mapping[str, Sequence[str]] | None = None) -> None:
        source = synonyms if synonyms is not None else DEFAULT_SYNONYMS
        self._synonyms = {key: tuple(values) for key, values in source.items()}

    def expand(self, query: str) -> str:
        """Return ``query`` with synonyms of its terms appended."""
        additions: list[str] = []
        seen = set(tokenize(query, remove_stopwords=False))
        for term in tokenize(query):
            for synonym in self._synonyms.get(term, ()):
                if synonym not in seen:
                    additions.append(synonym)
                    seen.add(synonym)
        if not additions:
            return query
        return f"{query} {' '.join(additions)}"


def expand_with_prf(
    query: str,
    retriever: Retriever,
    *,
    top_docs: int = 3,
    expansion_terms: int = 3,
) -> str:
    """Expand a query via pseudo-relevance feedback.

    Runs an initial retrieval, counts content terms across the top documents (excluding
    terms already in the query), and appends the most frequent ones.

    Args:
        query: The original query.
        retriever: An indexed retriever used for the initial pass.
        top_docs: How many top results to harvest terms from.
        expansion_terms: How many new terms to append.
    """
    if expansion_terms < 1:
        raise ValueError("expansion_terms must be at least 1")
    results = retriever.retrieve(query, k=top_docs)
    if not results:
        return query
    query_terms = set(tokenize(query))
    counts: Counter[str] = Counter()
    for scored in results:
        for term in tokenize(scored.chunk.text):
            if term not in query_terms:
                counts[term] += 1
    additions = [term for term, _ in counts.most_common(expansion_terms)]
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"
