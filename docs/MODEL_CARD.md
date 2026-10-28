# System Card - Hybrid RAG

## Overview

- **What it is:** a from-scratch hybrid retrieval system - BM25, a trained TF-IDF dense
  retriever, reciprocal rank fusion, and reranking - with a retrieval evaluation suite.
- **What it is not:** a generator. This is the retrieval half of a RAG system; it returns
  ranked passages, not generated answers.
- **Version:** 0.1.0.

## Intended use

- **Primary use cases:** learning how hybrid retrieval and its evaluation work; a
  from-scratch reference for BM25, TF-IDF, rank fusion, and IR metrics; a measurement
  harness for comparing retriever configurations.
- **Primary users:** engineers and researchers building or studying retrieval systems.

## Out-of-scope / prohibited use

- Not tuned for production-scale corpora; the index is in-memory.
- The bundled corpus is a demonstration, not a benchmark; do not quote its numbers as
  competitive retrieval results.

## Performance

See the ablation in the README and [EVALUATION_REPORT.md](EVALUATION_REPORT.md). On the
bundled corpus every pipeline reaches recall@3 = 1.0 and MRR about 0.97.

## Risks and limitations

- The dense retriever is lexical (TF-IDF); it misses zero-overlap paraphrases.
- In-memory index does not scale to very large corpora without a persistent store.

## Safety mitigations

- Deterministic, offline, reproducible; no network calls and no external data.

## Environmental / cost footprint

Indexing and evaluating the bundled corpus runs on CPU in well under a second, at no cost.
