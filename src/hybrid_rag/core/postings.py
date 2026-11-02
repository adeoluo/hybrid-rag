"""Compact inverted-index postings and exact top-k selection.

All postings live in two flat arrays, compressed sparse row style: chunk ids (``int32``) and
a per-entry weight (``float64``), grouped by term, with each term's ``[start, end)`` range in a
dict. That is about 12 bytes per entry instead of a Python tuple per entry, and a term's
postings are array slices (views, no copying).

Scoring a query adds each matching term's weights into one score array, term by term in query
order, so every chunk's score is the same sum, in the same order, as adding the weights one at
a time in Python; the vector adds just do it without an interpreter step per posting.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

Ids = npt.NDArray[np.int32]
Weights = npt.NDArray[np.float64]


class PostingsBuilder:
    """Collects ``(term, chunk id, value)`` entries, appended in increasing chunk order."""

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self._terms = array("i")
        self._ids = array("i")
        self._values = array("d")

    def add(self, term: str, chunk_id: int, value: float) -> None:
        self.add_chunk(chunk_id, [(term, value)])

    def add_chunk(self, chunk_id: int, items: Iterable[tuple[str, float]]) -> None:
        """Add every ``(term, value)`` of one chunk at once."""
        vocabulary = self.vocabulary
        setdefault = vocabulary.setdefault
        start = len(self._terms)
        for term, value in items:
            self._terms.append(setdefault(term, len(vocabulary)))
            self._values.append(value)
        self._ids.extend([chunk_id] * (len(self._terms) - start))

    def build(self) -> Postings:
        terms = np.frombuffer(self._terms, dtype=np.int32)
        order = np.argsort(terms, kind="stable")  # stable: chunk ids stay ascending per term
        counts = np.bincount(terms, minlength=len(self.vocabulary))
        offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        return Postings(
            vocabulary=self.vocabulary,
            offsets=offsets,
            term_of_entry=terms[order].copy(),
            ids=np.frombuffer(self._ids, dtype=np.int32)[order].copy(),
            values=np.frombuffer(self._values, dtype=np.float64)[order].copy(),
        )


class Postings:
    """Frozen postings: ``get(term)`` returns ``(chunk ids, weights)`` views, or ``None``."""

    def __init__(
        self,
        *,
        vocabulary: dict[str, int],
        offsets: npt.NDArray[np.int64],
        term_of_entry: Ids,
        ids: Ids,
        values: Weights,
    ) -> None:
        self.vocabulary = vocabulary
        self.offsets = offsets
        self.term_of_entry = term_of_entry
        self.ids = ids
        self.values = values

    @classmethod
    def empty(cls) -> Postings:
        return PostingsBuilder().build()

    def __len__(self) -> int:
        return len(self.vocabulary)

    def document_frequencies(self) -> npt.NDArray[np.int64]:
        """Number of chunks containing each term, indexed by vocabulary id."""
        return np.diff(self.offsets)

    def get(self, term: str) -> tuple[Ids, Weights] | None:
        index = self.vocabulary.get(term)
        if index is None:
            return None
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return self.ids[start:end], self.values[start:end]


def top_k(scores: Weights, k: int) -> list[tuple[int, float]]:
    """The ``k`` highest positive scores as ``(chunk id, score)``, best first.

    Equal scores keep chunk order, exactly like a stable descending sort over all chunks.
    """
    if k <= 0:
        return []
    docs = np.flatnonzero(scores > 0.0)
    values = scores[docs]
    if len(docs) > k:
        kth = np.partition(values, len(values) - k)[len(values) - k]  # the k-th largest
        above = np.flatnonzero(values > kth)
        tied = np.flatnonzero(values == kth)[: k - len(above)]
        chosen = np.sort(np.concatenate([above, tied]))
        docs, values = docs[chosen], values[chosen]
    order = np.lexsort((docs, -values))
    return [(int(docs[i]), float(values[i])) for i in order]
