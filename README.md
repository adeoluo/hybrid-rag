<h1 align="center">Hybrid RAG</h1>

<p align="center">
  <em>Production hybrid retrieval built from scratch: BM25 and a trained TF-IDF dense retriever, fused and reranked, with a rigorous retrieval evaluation suite.</em>
</p>

<p align="center">
  <a href="https://github.com/adeoluo/hybrid-rag/actions/workflows/ci.yml"><img alt="Continuous integration" src="https://github.com/adeoluo/hybrid-rag/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Coverage" src="https://img.shields.io/badge/coverage-95%25-brightgreen">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue">
  <img alt="Type checked" src="https://img.shields.io/badge/mypy-strict-blue">
  <img alt="Runtime dependencies" src="https://img.shields.io/badge/runtime%20deps-0-brightgreen">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## Motivation

Most retrieval-augmented generation tutorials import an embedding model and a vector
database and call it done. This project builds the retrieval stack **from first
principles** so the mechanics are visible: BM25's ranking function, a dense retriever that
*learns* its vector space from the corpus, rank fusion, reranking, and - most importantly -
the **evaluation** that tells you whether any of it works. It has **zero runtime
dependencies** and runs fully offline.

The thesis the project is built around: a retriever is only as good as your ability to
measure it. Every component is scored with standard information-retrieval metrics on a
labeled query set, and the pipelines are compared head to head in an ablation.

## Architecture

<p align="center">
  <img src="docs/architecture/architecture.svg" alt="System architecture" width="860">
</p>

Documents are split by a **chunker**; each chunk is indexed by **BM25** (sparse, lexical)
and a **TF-IDF dense** retriever (which learns its idf weights from the corpus). A query is
run through both; their rankings are merged by **reciprocal rank fusion**; a **reranker**
re-scores the fused candidates; and the whole pipeline is scored by the **evaluation
harness** (recall@k, precision@k, MRR, nDCG). See the [ADRs](docs/adr/) for key choices.

## Components

- **Chunking** - `FixedTokenChunker` (overlapping windows) and `SentenceChunker`.
- **BM25** - Okapi BM25 with configurable `k1` and `b`, implemented directly.
- **Dense** - a TF-IDF retriever that learns its vocabulary and idf from the corpus and
  ranks by cosine similarity (the "trained embedding").
- **Hybrid** - reciprocal rank fusion of any set of retrievers (combines ranks, not
  incomparable scores).
- **Reranking** - a cross-encoder-style `LexicalOverlapReranker` and a `RerankedRetriever`
  that composes retrieve-then-rerank into one retriever.
- **Evaluation** - recall@k, precision@k, MRR, MAP, DCG/nDCG, and an aggregating harness.

## Results

> Reproduce with `python -m hybrid_rag ablate`. Metrics averaged over 15 labeled queries.

| Pipeline | recall@3 | precision@3 | MRR | nDCG@3 |
|---|---:|---:|---:|---:|
| `bm25` | 1.000 | 0.611 | 0.967 | 0.975 |
| `dense` | 1.000 | 0.611 | 0.967 | 0.975 |
| `hybrid` | 1.000 | 0.611 | 0.967 | 0.975 |
| `reranked` | 1.000 | 0.611 | 0.967 | 0.975 |

On this clean, lexically-rich corpus the pipelines perform comparably: each finds the
relevant passage, and the harness confirms it with calibrated metrics. The point the
numbers make is methodological - **every pipeline is measured identically**, so a genuine
improvement (a better reranker, real neural embeddings) would show up immediately as a
metric gain. See [limitations](#limitations) for why lexical methods converge here.

## Quickstart

```bash
git clone https://github.com/adeoluo/hybrid-rag.git && cd hybrid-rag
make setup                    # install dev tooling + pre-commit hooks
make all                      # lint + type-check + test

# evaluate every pipeline on the bundled labeled queries
python -m hybrid_rag ablate --k 3

# retrieve over the bundled corpus
python -m hybrid_rag query "how do plants make energy from sunlight" --pipeline hybrid
```

Use the library directly:

```python
from hybrid_rag.core import FixedTokenChunker, chunk_documents, Document
from hybrid_rag.retrievers import BM25Retriever, TfidfDenseRetriever, HybridRetriever

chunks = chunk_documents([Document(id="d", text="...")], FixedTokenChunker())
retriever = HybridRetriever([BM25Retriever(), TfidfDenseRetriever()])
retriever.index(chunks)
results = retriever.retrieve("my query", k=5)
```

## Design decisions

- **Reciprocal rank fusion, not score addition.** BM25 and cosine scores are on
  incomparable scales; fusing ranks is robust and parameter-light.
  ([ADR-0002](docs/adr/0002-reciprocal-rank-fusion.md))
- **Learn the dense space from the corpus.** TF-IDF idf weights are fit at index time, so
  the dense retriever is genuinely trained, not pretrained-and-imported.
  ([ADR-0003](docs/adr/0003-tfidf-dense-retriever.md))
- **Evaluation is a first-class component**, not an afterthought: the project ships the
  metrics and the harness, and every claim is a reproducible number.

## Responsible AI

- **Model card:** [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- **Data card:** [docs/DATA_CARD.md](docs/DATA_CARD.md)
- **Threat model:** [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md)

## Limitations

- The dense retriever is TF-IDF, which shares vocabulary with BM25; it cannot match
  paraphrases with no lexical overlap. True semantic retrieval needs neural embeddings
  (a deliberate boundary to keep the project dependency-free and from-scratch).
- Because both retrievers are lexical, they converge on the clean bundled corpus; the
  value here is the measurement framework, which would expose differences on a harder or
  semantically-phrased query set.
- The bundled corpus is a demonstration, not a competitive benchmark.

## Roadmap

- [ ] Optional neural embedding retriever behind the same `Retriever` interface
- [ ] Query expansion and pseudo-relevance feedback
- [ ] Persistent on-disk index for larger corpora

## License

[MIT](LICENSE). Part of a 28-project artificial intelligence engineering portfolio by
[adeoluo](https://github.com/adeoluo).
