# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## Unreleased

### Documentation

- Standard README structure and language framing (#2)
- Horizontal layered architecture diagram
- Compact horizontal architecture diagram

### Build and CI

- Run CodeQL on main and on a weekly schedule only

## [0.1.0] 2026-11-02

### Features

- **core:** Add retrieval types, tokenization, and chunking
- **retrievers:** Add BM25 and a trained TF-IDF dense retriever
- **fusion:** Add hybrid reciprocal-rank fusion and reranking
- **evaluation:** Add retrieval metrics and an evaluation harness
- **cli:** Add bundled dataset, pipelines, and an ablation CLI
- **query:** Add query expansion and advanced chunking
- **retrievers:** Add char-ngram and phrase retrievers and MMR reranking
- **rag:** Add cited context assembly and chunk persistence
- **evaluation:** Add metrics, retriever comparison, and new pipelines
- Add HTTP service, stress benchmark, and operations documentation
- Add production observability (metrics and readiness) and reword footer
- From-scratch neural dense retriever (#1)

### Bug fixes

- Describe the TF-IDF retriever accurately, not as a dense embedding (#20)
- Precision@k must divide by k, not by results returned (#21)

### Documentation

- Write README, architecture diagram, and responsible-AI cards
- Add performance benchmark and refresh the README
- Use plain, factual wording (#19)

### Maintenance

- Scaffold hybrid-rag from standards template
- Format benchmark
- **deps:** Bump actions/checkout from 4 to 7
- **deps:** Bump astral-sh/setup-uv from 4 to 7
- **deps:** Bump actions/upload-artifact from 4 to 7
- **ci:** Bump GitHub Actions to current major versions
- Remove Dependabot configuration

### Bench

- Add large-workload scale benchmark (#22)
