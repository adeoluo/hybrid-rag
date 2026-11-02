<h1 align="center">Hybrid RAG</h1>

<p align="center">
  <em>Hybrid retrieval built from scratch: stemmed BM25 and TF-IDF over inverted indexes, fused and reranked, evaluated on the BEIR benchmark.</em>
</p>

<p align="center">
  <a href="https://github.com/adeoluo/hybrid-rag/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/adeoluo/hybrid-rag/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/adeoluo/hybrid-rag/releases"><img alt="Release" src="https://img.shields.io/github/v/release/adeoluo/hybrid-rag"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

---

## Motivation

Most retrieval-augmented generation tutorials import an embedding model and a vector
database and call it done. This project builds the retrieval stack **from scratch** so the
mechanics are visible: BM25's ranking function, a TF-IDF vector retriever fit from the corpus,
rank fusion, reranking, and, most importantly,
the **evaluation** that tells you whether any of it works. It runs fully offline.

The thesis the project is built around: a retriever is only as good as your ability to
measure it. Every component is scored with standard information-retrieval metrics on a
labeled query set, and the pipelines are compared head to head in an ablation.

## Architecture

<p align="center">
  <img src="docs/architecture/architecture.svg" alt="Architecture diagram" width="860">
</p>

<sub>Legend: amber = entry points and external systems, indigo = core logic, green = data and state, blue = validation and control, grey = observability and outputs.</sub>

Documents are split by a **chunker** and indexed once per named pipeline. The sparse
retrievers are **BM25**, **TF-IDF**, character n-grams, and exact phrase matching; a
from-scratch **neural dense retriever** ranks by cosine similarity over skip-gram word
embeddings. A query, from the CLI or a validated `/retrieve` request, runs through the chosen
pipeline: a single retriever, or several whose rankings are merged by **reciprocal rank
fusion**, optionally followed by a **reranker** (lexical overlap or MMR). The **evaluation
harness** scores every pipeline on labeled queries (recall@k, precision@k, MRR, nDCG), a paired
bootstrap compares two pipelines, and every route records latency and errors for `/metrics`.
See the [ADRs](docs/adr/) for key choices.

### Components

- **Chunking**: five strategies: `FixedTokenChunker` (overlapping windows),
  `SentenceChunker`, `RecursiveChunker` (separator priority), and `ParagraphChunker`.
- **Retrievers (5)**: Okapi `BM25Retriever`, a `TfidfRetriever`, a
  typo-robust `CharNgramRetriever` (character n-grams), an exact-match `PhraseRetriever`,
  and the fusing `HybridRetriever`.
- **Query processing**: `SynonymExpander` and pseudo-relevance feedback (`expand_with_prf`)
  to improve recall before retrieval runs.
- **Hybrid fusion**: reciprocal rank fusion of any set of retrievers (combines ranks, not
  incomparable scores).
- **Reranking**: a cross-encoder-style `LexicalOverlapReranker`, an `MMRReranker` for
  diversity, and a `RerankedRetriever` that composes retrieve-then-rerank into one retriever.
- **RAG**: `build_context` assembles cited, budget-bounded context with snippet
  highlighting; `ChunkStore` persists a chunked corpus to disk.
- **Evaluation**: recall@k, precision@k, MRR, MAP, DCG/nDCG, success@k, R-precision, an
  aggregating harness, and `compare_retrievers` with a paired-bootstrap significance test.

## Results

> Reproduce with `python -m hybrid_rag ablate`. Metrics averaged over 15 labeled queries at k=5.

| Pipeline | recall@5 | precision@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| `bm25` | 1.000 | 0.200 | 0.933 | 0.951 |
| `tfidf` | 1.000 | 0.200 | 0.933 | 0.951 |
| `dense` | 1.000 | 0.200 | 0.933 | 0.951 |
| `hybrid` | 1.000 | 0.200 | 0.933 | 0.951 |
| `dense-hybrid` | 1.000 | 0.200 | 0.933 | 0.951 |
| `reranked` | 1.000 | 0.200 | 0.933 | 0.951 |
| `hybrid-all` | 1.000 | 0.200 | 0.933 | 0.951 |
| `mmr` | 1.000 | 0.200 | 0.913 | 0.935 |

The pipelines are measured identically on the same labeled queries. Each query has a
single relevant chunk, so precision@5 is capped at 0.200 (one correct result out of five
positions) and is identical across pipelines; it is reported for completeness rather than as
a differentiator on this corpus. Every pipeline recovers the relevant chunk (recall 1.000),
so the differences that remain are in **ranking quality** (MRR and nDCG). This set is too small
to separate the methods: Porter stemming, which raises BM25's nDCG@10 on the real BEIR datasets
below, moves one of these 15 queries down a rank (MRR 0.967 before stemming, 0.933 after), which
is why the real-data benchmark is the one to judge by.

### Real benchmark: BEIR

> Reproduce: download a BEIR dataset (for example
> `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip`), unzip it, and run
> `uv run --with rank_bm25 python benchmarks/beir.py --data scifact --rank-bm25`.
> nDCG@10 with graded gains and capped recall@100, as in BEIR. Apple M1, one thread. Raw numbers: `benchmarks/results/`.

[BEIR](https://github.com/beir-cellar/beir) is the standard zero-shot retrieval benchmark, so these
numbers are comparable with published ones. The published BEIR BM25 baseline (Elasticsearch) is
**0.665** nDCG@10 on SciFact and **0.656** on TREC-COVID. `rank_bm25` (BM25Okapi) is run on the same
tokens as an independent implementation.

**SciFact**: 5,183 scientific abstracts, 300 queries.

| Retriever | nDCG@10 | Recall@100 | Index build | Query |
|---|---:|---:|---:|---:|
| **bm25** | **0.687** | **0.928** | 0.56 s | **0.18 ms** |
| tfidf | 0.606 | 0.926 | 0.56 s | 0.18 ms |
| hybrid | 0.665 | 0.926 | 0.97 s | 0.88 ms |
| reranked | 0.596 | 0.865 | 1.21 s | 1.60 ms |
| hybrid-all | 0.615 | 0.927 | 5.92 s | 2.99 ms |
| mmr | 0.657 | 0.865 | 0.95 s | 10.38 ms |
| rank_bm25 BM25Okapi | 0.665 | 0.886 | 0.45 s | 13.19 ms |

**TREC-COVID**: 171,331 CORD-19 papers, 50 queries with deep judgments. Peak memory for the
whole run (corpus, all three indexes and rank_bm25) is 1.75 GB.

| Retriever | nDCG@10 | Recall@100 | Index build | Query |
|---|---:|---:|---:|---:|
| **bm25** | **0.663** | **0.528** | 20.9 s | **2.5 ms** |
| tfidf | 0.472 | 0.424 | 18.6 s | 2.1 ms |
| hybrid | **0.666** | 0.522 | 34.0 s | 5.1 ms |
| rank_bm25 BM25Okapi | 0.578 | 0.471 | 10.0 s | 514.2 ms |

The machine had other applications running, so the absolute times are noisy; rows in one table
come from the same run and compare fairly.

What changed and what it bought:

- **Porter stemming**, written from scratch in `core/stemming.py`. It gives the same stems as
  NLTK's original-algorithm Porter stemmer on all 34,955 SciFact words longer than two letters.
  It lifts BM25 from 0.669 to **0.687** nDCG@10 on SciFact and recall@100 from 0.886 to 0.928.
  BM25 now **beats the published BEIR baseline on SciFact (0.687 against 0.665) and matches it on
  TREC-COVID (0.663 against 0.656)**, and beats `rank_bm25` on the same tokens by 0.022 and 0.085.
- **Inverted indexes in flat arrays** for BM25, TF-IDF, character n-grams and phrases. Each
  term's postings are slices of two numpy arrays (chunk ids and precomputed weights), and a query
  adds the weights of its terms into one score array, so only chunks that share a term with the
  query are touched. With stemming off they return exactly the rankings of the old full scans
  (checked on every SciFact query; BM25, character n-gram and phrase scores are bit for bit
  equal). Queries got **44x faster for BM25 on SciFact** (7.98 ms to 0.18 ms), 19x for `hybrid`
  and 27x for `hybrid-all`, and BM25 answers **about 200x faster than `rank_bm25` on
  TREC-COVID**. The hybrid pipeline on TREC-COVID went from 731 ms to 5.1 ms per query: the old
  per-posting Python tuples pushed the 171,000-document indexes into swap on this 8 GB machine.

Where it does not win:

- **Fusion does not help on these datasets.** The `hybrid` pipeline is slightly worse than BM25
  alone on SciFact (0.665) and only ties it on TREC-COVID (0.666), because its second retriever,
  TF-IDF, is a weaker lexical signal rather than a different one. The lexical reranker and MMR
  hurt nDCG. A neural dense retriever trained on far more data is what fusion needs to pay off.
- **Index build is slower than `rank_bm25`** (20.9 s against 10.0 s on TREC-COVID), mostly
  because every word is stemmed and every posting is written out; the build runs in Python.
- **Every result is lexical.** Learned dense retrievers and cross-encoders score higher on both
  datasets in the BEIR leaderboard; this library does not include one.

### What the neural dense retriever shows (an honest result)

The `dense` pipeline is a **from-scratch neural retriever**: it trains word embeddings with a
skip-gram objective and negative sampling over the indexed corpus (with the training loop and
gradients written by hand), represents each chunk and query as the mean of its word vectors, and
ranks by cosine similarity. It is the "real neural embeddings" baseline the lexical methods are measured
against.

The honest finding is that on this small corpus the learned dense retriever **does not beat the
lexical baselines**: it recovers every relevant chunk and ties them (MRR 0.933), and before the
lexical retrievers gained stemming it trailed them (0.933 against 0.967). Learning dense semantics
that outranks exact lexical matching needs far more training data than a few short passages
provide, where a single shared term is a very strong and precise signal. Fusing it with the
lexical ones (`dense-hybrid`) does not change the ranking here. The value here is a
correct, measured, from-scratch implementation of dense retrieval, compared fairly against
lexical retrieval rather than assumed to beat it. See [limitations](#limitations) for why the
methods converge at this corpus scale.

### Scale

`benchmarks/scale.py` runs the flagship `hybrid` pipeline (BM25 + TF-IDF fusion)
over progressively larger synthetic chunk corpora and a fixed 80-query set,
reporting wall-clock time, query throughput, peak memory (stdlib `tracemalloc`,
measured in a separate pass so the per-allocation hook does not distort timing),
and recall@5 against one planted relevant chunk per query. Measured on this
machine (`uv run python benchmarks/scale.py`):

| Chunks | Seconds | Queries/sec | Peak MiB | recall@5 |
|---:|---:|---:|---:|---:|
| 500 | 0.025 | 8,431 | 0.8 | 0.53 |
| 1,000 | 0.046 | 6,308 | 1.6 | 0.38 |
| 2,000 | 0.076 | 6,168 | 3.1 | 0.31 |
| 4,000 | 0.139 | 4,209 | 6.2 | 0.21 |

Before the inverted indexes the same runs took 1.35 to 10.07 seconds at 67 to 9 queries per
second with 2.3 to 18.3 MiB peak; throughput now falls slowly because a query only touches
chunks that share a term with it. Recall decays as the field of random distractors grows. These
are synthetic data; the BEIR section above is the real-data measurement.

### Performance

> Measured over a 500-chunk synthetic corpus and 200 queries (`uv run python benchmarks/benchmark.py`).

| Pipeline | Index build (ms) | Queries / sec |
|---|---:|---:|
| `bm25` | 8.4 | 23,443 |
| `tfidf` | 8.7 | 24,301 |
| `hybrid` | 15.4 | 7,608 |
| `reranked` | 14.9 | 2,906 |
| `hybrid-all` (4-way fusion) | 80.0 | 2,652 |
| `mmr` (diversity) | 14.7 | 951 |

All retrieval is in-process and in-memory. Against the previous full scans, query throughput is
8 to 14 times higher (BM25 1,696 to 23,443 queries per second); index build is a few
milliseconds slower because postings and stems are computed up front.

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
from hybrid_rag.retrievers import BM25Retriever, TfidfRetriever, HybridRetriever

chunks = chunk_documents([Document(id="d", text="...")], FixedTokenChunker())
retriever = HybridRetriever([BM25Retriever(), TfidfRetriever()])
retriever.index(chunks)
results = retriever.retrieve("my query", k=5)
```

## Usage

A real run: a hybrid query with fused scores, then the retrieval ablation across eight pipelines. Regenerate with `vhs docs/demo/query.tape`.

<p align="center">
  <img src="docs/demo/query.gif" alt="hybrid-rag query and retrieval ablation" width="860">
</p>

### HTTP service

A standard-library HTTP service indexes every pipeline once at startup and serves retrieval:

```bash
python -m hybrid_rag serve --port 8000

curl localhost:8000/health
curl -s -X POST localhost:8000/retrieve -d '{"query": "neural networks", "pipeline": "hybrid", "k": 5}'
```

`GET /health` lists the pipelines and chunk count; `POST /retrieve` returns ranked chunks
with scores for a chosen pipeline, so the latency-for-quality tradeoff is selectable per
request. A production-style stress benchmark (`uv run python benchmarks/stress.py`) reports
throughput and p50/p95/p99 latency per pipeline, making that tradeoff concrete; the service
is scaled with processes because the CPU-bound Python handler is limited by the global interpreter
lock. The service also exposes `GET /ready` (readiness) and `GET /metrics` (per-route request, error,
and latency metrics in Prometheus text format). See the [operations guide](docs/OPERATIONS.md), the
[scaling and tradeoffs](docs/SCALING.md) document, and the [case study](docs/CASE_STUDY.md) on choosing a pipeline under a budget.

### Demos

Recorded from real runs with [vhs](https://github.com/charmbracelet/vhs); each tape is in `docs/demo/`.

The same query through three retrieval pipelines.

<p align="center">
  <img src="docs/demo/pipelines.gif" alt="The same query through three retrieval pipelines" width="860">
</p>

The HTTP service retrieving with the hybrid pipeline.

<p align="center">
  <img src="docs/demo/http.gif" alt="The HTTP service retrieving with the hybrid pipeline" width="860">
</p>

## Design decisions

- **Reciprocal rank fusion, not score addition.** BM25 and cosine scores are on
  incomparable scales; fusing ranks is robust and parameter-light.
  ([ADR-0002](docs/adr/0002-reciprocal-rank-fusion.md))
- **Fit the TF-IDF weights from the corpus.** The idf weights are computed at index time,
  so the retriever is corpus-specific; it is a sparse lexical method, not a neural dense embedding.
  ([ADR-0003](docs/adr/0003-tfidf-retriever.md))
- **Inverted indexes in flat arrays, with exact parity.** Postings are two numpy arrays per
  retriever, grouped by term, and queries add precomputed weights into a score array. The scoring
  formulas are unchanged and the additions happen in the same order, so the rankings are the
  ones a full scan gives. ([ADR-0004](docs/adr/0004-inverted-index-and-stemming.md))
- **Porter stemming on by default for BM25 and TF-IDF**, because it raises nDCG@10 on real
  data (BEIR); `stem=False` restores exact-word matching.
- **Evaluation is a first-class component**, not an afterthought: the project ships the
  metrics and the harness, and every claim is a reproducible number.

## Testing

```bash
make test                    # uv run pytest
uv run pytest -m "not slow"   # the CI selection, without full-pipeline training runs
make lint type               # ruff format check, ruff lint, mypy
```

The suite in `tests/` covers tokenization and every chunker (`test_core.py`, `test_query_chunking.py`), each retriever including BM25, TF-IDF, character n-gram, phrase, and the from-scratch neural dense retriever (`test_retrievers.py`, `test_more_retrievers.py`, `test_dense.py`), reciprocal rank fusion, reranking, and MMR (`test_fusion_rerank.py`), the retrieval metrics and evaluation harness (`test_evaluation.py`), paired-bootstrap retriever comparison (`test_comparison.py`), RAG context assembly and chunk persistence (`test_rag_persistence.py`), the command-line interface (`test_cli.py`), and the HTTP handler layer (`test_service.py`).

Tests marked `slow` train every pipeline end to end, including the neural dense retriever; they are excluded from CI and reproduced by the CLI and benchmarks. CI runs ruff format, ruff lint, mypy, and `pytest -m "not slow"` with an 80% coverage gate on Python 3.11 and 3.12, then builds the package and the Docker image.

## Responsible AI

- **Model card:** [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- **Data card:** [docs/DATA_CARD.md](docs/DATA_CARD.md)
- **Threat model:** [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md)

## Limitations

- The TF-IDF retriever is sparse and lexical, sharing vocabulary with BM25; it cannot match
  paraphrases with no lexical overlap. True semantic retrieval needs neural embeddings
  (a deliberate boundary to keep every retriever from scratch and inspectable).
- Because both retrievers are lexical, they converge on the clean bundled corpus; the
  value here is the measurement framework, which would expose differences on a harder or
  semantically-phrased query set.
- The bundled corpus is a demonstration, not a competitive benchmark; the BEIR results are the
  real-data measurement.
- Indexing runs in Python, so building over 171,000 documents takes about 20 seconds per
  retriever, twice `rank_bm25`'s time, and every index lives in memory.

## Roadmap

- [ ] Optional neural embedding retriever behind the same `Retriever` interface
- [ ] Query expansion and pseudo-relevance feedback
- [ ] Persistent on-disk index for larger corpora

## License

Released under the [MIT License](LICENSE).
