# ADR-0003: A TF-IDF vector-space retriever fit from the corpus

- **Status:** Accepted
- **Date:** 2026-01-01

## Context

The project needs a second retriever to contrast with BM25, and it aims to be from scratch
and dependency-free. Importing a pretrained neural embedding model would add a heavy
dependency and make the representation a black box.

## Decision

Implement a TF-IDF vector-space retriever that fits its weights from the indexed corpus:
inverse-document-frequency weights are computed at index time, chunks and queries are
represented as sparse TF-IDF vectors, and ranking is by cosine similarity. This is a
classical, sparse, lexical method, not a neural dense embedding.

## Alternatives considered

- **Pretrained sentence embeddings.** Stronger on paraphrases, but a heavy dependency and
  opaque; out of scope for a from-scratch build (noted as a roadmap item).
- **Raw term-frequency vectors.** Simpler, but without idf, common words dominate and
  ranking quality drops.

## Consequences

- The retriever is corpus-specific (its idf is fit from the indexed corpus) and fully
  interpretable, but it is a sparse lexical representation, not a learned dense embedding.
- It shares vocabulary with BM25, so it cannot match zero-overlap paraphrases - an accepted
  limitation documented in the README. Bridging that gap is what a neural dense retriever
  adds.
- The `Retriever` interface is unchanged, so a neural embedding retriever can later drop in
  behind it without touching fusion, reranking, or evaluation.
