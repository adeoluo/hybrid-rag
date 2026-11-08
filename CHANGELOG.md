# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## [0.2.1] 2026-11-08

### Bug fixes

- Report the released version and keep version strings in step on release (#10)

### Documentation

- Describe the supply chain accurately (#7)
- Complete citation metadata and bump it on release (#9)

### Maintenance

- Latest toolchain and CI versions (#8)

## [0.2.0] 2026-11-02

### Performance

- Inverted indexes in flat arrays, Porter stemming, BEIR benchmark (#6)

### Documentation

- Standard README structure and language framing (#2)
- Horizontal layered architecture diagram
- Compact horizontal architecture diagram
- Recorded terminal demo of a real run, with its vhs tape (#3)
- Remove authorship statements from docs and scripts
- Recorded demos of more commands, with their vhs tapes (#4)
- Recorded demos of the HTTP service and more commands (#5)

### Build and CI

- Run CodeQL on main and on a weekly schedule only

### Maintenance

- Add release automation, AGENTS.md, llms.txt and contributor docs
- Classify squash merges in the changelog by the commits they contain

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
