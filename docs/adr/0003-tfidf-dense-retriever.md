# ADR-0003: A TF-IDF dense retriever learned from the corpus

- **Status:** Accepted
- **Date:** 2026-01-01

## Context

The project needs a dense retriever to contrast with BM25, and it aims to be from-scratch
and dependency-free. Importing a pretrained neural embedding model would add heavy
dependencies and make the "embedding" a black box.

## Decision

Implement a TF-IDF dense retriever that *learns* its vector space from the indexed corpus:
inverse-document-frequency weights are fit at index time, chunks and queries are embedded
as sparse TF-IDF vectors, and ranking is by cosine similarity.

## Alternatives considered

- **Pretrained sentence embeddings.** Stronger on paraphrases, but a heavy dependency and
  opaque; out of scope for a from-scratch build (noted as a roadmap item).
- **Raw term-frequency vectors.** Simpler, but without idf, common words dominate and
  ranking quality drops.

## Consequences

- The dense retriever is genuinely trained on the corpus and fully interpretable.
- It shares vocabulary with BM25, so it cannot match zero-overlap paraphrases - an
  accepted limitation documented in the README.
- The `Retriever` interface is unchanged, so a neural embedding retriever can later drop
  in behind it without touching fusion, reranking, or evaluation.
