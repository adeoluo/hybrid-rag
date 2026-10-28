# ADR-0002: Reciprocal rank fusion for hybrid retrieval

- **Status:** Accepted
- **Date:** 2026-01-01

## Context

The hybrid retriever must combine the rankings of a sparse (BM25) and a dense (TF-IDF)
retriever. Their relevance scores are on completely different, uncalibrated scales, so
adding or averaging the scores is meaningless.

## Decision

Combine the retrievers with reciprocal rank fusion (RRF): each chunk's fused score is the
sum over retrievers of `1 / (k + rank)`, where `rank` is its position in that retriever's
list and `k` is a smoothing constant (default 60).

## Alternatives considered

- **Weighted score addition.** Requires normalizing incomparable score distributions and
  tuning weights per corpus; brittle.
- **Learning to rank.** More powerful, but needs training data and adds machinery out of
  proportion to the goal.

## Consequences

- Fusion depends only on ranks, so it is robust to each retriever's score scale.
- A single, intuitive parameter (`k`) controls how sharply top ranks are rewarded.
- Any number of retrievers can be fused through the same mechanism.
