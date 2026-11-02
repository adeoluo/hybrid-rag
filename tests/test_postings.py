"""Exact top-k selection over score arrays."""

from __future__ import annotations

import random

import numpy as np
import pytest

from hybrid_rag.core.postings import Postings, PostingsBuilder, top_k


def reference(scores: list[float], k: int) -> list[tuple[int, float]]:
    ranked = [(doc, s) for doc, s in enumerate(scores) if s > 0.0]
    ranked.sort(key=lambda item: item[1], reverse=True)  # stable: ties keep chunk order
    return ranked[:k]


@pytest.mark.parametrize("seed", range(20))
def test_matches_a_stable_sort_with_many_ties(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 300)
    # Few distinct values, so ties straddle the k-th position often.
    scores = [rng.choice([0.0, 0.0, 0.5, 1.0, 1.5, 2.0, -1.0]) for _ in range(n)]
    for k in (0, 1, 5, 17, n, n + 3):
        assert top_k(np.asarray(scores, dtype=np.float64), k) == reference(scores, k)


def test_empty_and_non_positive() -> None:
    assert top_k(np.zeros(0), 5) == []
    assert top_k(np.asarray([0.0, -2.0]), 5) == []


def test_postings_group_entries_by_term_in_chunk_order() -> None:
    builder = PostingsBuilder()
    for term, doc, value in [("cat", 0, 1.0), ("dog", 0, 2.0), ("cat", 3, 2.5), ("dog", 4, 0.5)]:
        builder.add(term, doc, value)
    postings = builder.build()
    ids, weights = postings.get("cat")  # type: ignore[misc]
    assert ids.dtype == np.int32
    assert weights.dtype == np.float64
    assert ids.tolist() == [0, 3]
    assert weights.tolist() == [1.0, 2.5]
    assert postings.get("dog")[0].tolist() == [0, 4]  # type: ignore[index]
    assert postings.get("bird") is None
    assert postings.document_frequencies().tolist() == [2, 2]
    assert len(postings) == 2
    assert len(Postings.empty()) == 0
