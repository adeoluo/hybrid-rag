# AGENTS.md

Working notes for coding agents and contributors in **Hybrid RAG**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

Production hybrid retrieval built from scratch: BM25 and a TF-IDF vector retriever, fused and reranked, with a rigorous retrieval evaluation suite.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docker-compose.yml`: local container setup
- `docs/`: documentation, architecture diagram source and render, design records
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `src/hybrid_rag/__init__.py`: Hybrid RAG - production hybrid retrieval built from scratch.
- `src/hybrid_rag/cli.py`: Command-line interface for the hybrid retrieval system.
- `src/hybrid_rag/core/__init__.py`: Core abstractions for hybrid retrieval.
- `src/hybrid_rag/core/chunking.py`: Chunking strategies: splitting documents into retrievable spans.
- `src/hybrid_rag/core/chunking_advanced.py`: Additional chunking strategies.
- `src/hybrid_rag/core/retriever.py`: The retriever interface.
- `src/hybrid_rag/core/tokenization.py`: Text tokenization shared across retrievers.
- `src/hybrid_rag/core/types.py`: Core data types for retrieval.
- `src/hybrid_rag/data/sample.py`: A small bundled corpus and labeled query set for reproducible evaluation.
- `src/hybrid_rag/evaluation/__init__.py`: Retrieval evaluation: metrics and a harness.
- `src/hybrid_rag/evaluation/comparison.py`: Additional retrieval metrics and a bootstrap comparison of two retrievers.
- `src/hybrid_rag/evaluation/harness.py`: A retrieval evaluation harness.
- `src/hybrid_rag/evaluation/metrics.py`: Retrieval quality metrics, implemented from scratch.
- `src/hybrid_rag/mmr.py`: Maximal Marginal Relevance (MMR) diversity reranking.
- `src/hybrid_rag/observability.py`: In-process request metrics for the HTTP service.
- `src/hybrid_rag/persistence.py`: Persisting a chunked corpus to disk.
- `src/hybrid_rag/pipelines.py`: Convenience builders for the named retriever configurations.
- `src/hybrid_rag/query.py`: Query processing: expansion and normalization.
- `src/hybrid_rag/rag.py`: Retrieval-augmented generation: assembling cited context for a reader.
- `src/hybrid_rag/reranking.py`: Reranking: a second-stage scorer over a first-stage candidate set.
- `src/hybrid_rag/retrievers/__init__.py`: Retrievers: sparse (BM25), dense (TF-IDF), and hybrid fusion.
- `src/hybrid_rag/retrievers/bm25.py`: BM25 lexical retriever, implemented from scratch.
- `src/hybrid_rag/retrievers/char_ngram.py`: Character n-gram retriever, robust to typos and morphology.
- `src/hybrid_rag/retrievers/dense.py`: A from-scratch neural dense retriever.
- `src/hybrid_rag/retrievers/hybrid.py`: Hybrid retrieval via reciprocal rank fusion.
- `src/hybrid_rag/retrievers/phrase.py`: Exact phrase retriever.
- `src/hybrid_rag/retrievers/tfidf.py`: A TF-IDF vector-space retriever that fits its weights from the corpus.
- `src/hybrid_rag/service.py`: A standard-library HTTP service exposing hybrid retrieval.

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make evaluation` | Reproduce the results table |
| `make run` | Run the application entrypoint |
| `make diagram` | Render architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make docker-up` | Run via docker compose |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut with `make cut-release`.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.
