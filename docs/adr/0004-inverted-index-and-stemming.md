# ADR-0004: Inverted indexes in flat arrays, and Porter stemming

- **Status:** Accepted
- **Date:** 2026-11-02

## Context

Every retriever scored a query by scanning all chunks in Python. On BEIR SciFact (5,183
abstracts) BM25 took about 8 ms per query; on TREC-COVID (171,331 papers) the hybrid pipeline
took about 0.7 s, because postings stored as Python tuples pushed the indexes into swap on an
8 GB machine. Retrieval quality on BEIR also trailed the published BM25 baseline on SciFact
partly because "treatments" and "treated" did not match.

## Decision

- Index BM25, TF-IDF, character n-grams and phrases with inverted postings held in two flat
  numpy arrays (chunk ids as `int32`, weights as `float64`), grouped by term, with each term's
  range in a dict (`core/postings.py`). BM25's per-chunk term weight is precomputed at index
  time. A query adds the weights of its terms into one score array; top-k uses a partition and
  breaks ties by chunk order, like a stable sort.
- Keep the scoring formulas and the order of additions unchanged, so rankings equal those of the
  previous full scans. This is checked against the previous implementation on every SciFact query.
- Add a Porter stemmer (`core/stemming.py`), on by default for BM25 and TF-IDF.

## Alternatives considered

- **Keep the full scans.** Simple, but query cost grows with the corpus and the memory layout does
  not fit large corpora.
- **WAND or MaxScore early termination.** Faster still for long postings, but more code; the
  array accumulation is already a few milliseconds per query at 171,000 documents.
- **A stemmer library such as NLTK.** Adds a dependency for one function; the algorithm is short
  and is checked against NLTK's output.

## Consequences

- Queries are 19 to 44 times faster on SciFact and about 200 times faster than `rank_bm25` on
  TREC-COVID; BM25 beats the published BEIR baseline on SciFact and matches it on TREC-COVID.
- numpy becomes a runtime dependency.
- Index build does more work up front (stemming, writing postings), so it is about twice as slow
  as `rank_bm25`'s build.
- Stemming changes rankings; `stem=False` gives exact-word matching.
