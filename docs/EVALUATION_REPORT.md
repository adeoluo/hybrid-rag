# Evaluation Report - Hybrid RAG

> Reproduce real-corpus benchmarks with `uv run --with rank_bm25 python benchmarks/beir.py --data scifact --rank-bm25`
> (raw outputs in `benchmarks/results/`) and the offline CI smoke fixture with `python -m hybrid_rag ablate`.

## 1. Zero-shot retrieval on BEIR (SciFact and TREC-COVID)

- **Datasets:** BEIR **SciFact** (5,183 scientific abstracts, 300 queries) and **TREC-COVID**
  (171,331 CORD-19 papers, 50 queries with deep graded relevance judgments).
- **Metrics:** nDCG@10 with graded gains and capped recall@100, matching the official BEIR
  protocol.
- **Baselines:** published BEIR BM25 (Elasticsearch: `0.665` on SciFact, `0.656` on TREC-COVID)
  plus `rank_bm25` (`BM25Okapi`) run in-process on identical tokens.

### SciFact (5,183 abstracts, 300 queries)

| Retriever | nDCG@10 | Recall@100 | Index build | Query |
|---|---:|---:|---:|---:|
| **bm25** (Porter stemmer + flat NumPy postings) | **0.687** | **0.928** | 0.56 s | **0.18 ms** |
| tfidf | 0.606 | 0.926 | 0.56 s | 0.18 ms |
| hybrid (BM25 + TF-IDF RRF) | 0.665 | 0.926 | 0.97 s | 0.88 ms |
| reranked (hybrid + lexical overlap) | 0.596 | 0.865 | 1.21 s | 1.60 ms |
| hybrid-all | 0.615 | 0.927 | 5.92 s | 2.99 ms |
| mmr | 0.657 | 0.865 | 0.95 s | 10.38 ms |
| `rank_bm25` (`BM25Okapi`) | 0.665 | 0.886 | 0.45 s | 13.19 ms |

### TREC-COVID (171,331 papers, 50 queries)

| Retriever | nDCG@10 | Recall@100 | Index build | Query |
|---|---:|---:|---:|---:|
| **bm25** | **0.663** | **0.528** | 20.9 s | **2.5 ms** |
| tfidf | 0.472 | 0.424 | 18.6 s | 2.1 ms |
| **hybrid** (BM25 + TF-IDF RRF) | **0.666** | 0.522 | 34.0 s | 5.1 ms |
| `rank_bm25` (`BM25Okapi`) | 0.578 | 0.471 | 10.0 s | 514.2 ms |

### Key findings and honest limitations

- **From-scratch Porter stemming + flat NumPy inverted indexes** beat published BEIR Elasticsearch
  BM25 on SciFact (`0.687` vs `0.665`) and match on TREC-COVID (`0.663` vs `0.656`), while
  answering ~200x faster than `rank_bm25` on TREC-COVID (`2.5 ms` vs `514.2 ms`).
- **Lexical-with-lexical RRF fusion does not beat BM25 alone on SciFact (`0.665` vs `0.687`)**
  and only ties on TREC-COVID (`0.666` vs `0.663`), because TF-IDF is a weaker lexical signal
  rather than an orthogonal semantic representation. Neural dense embeddings or cross-encoders are
  required for hybrid fusion to yield large gains.

---

## 2. Offline CI regression smoke fixture (`python -m hybrid_rag ablate`)

For sub-second hermetic CI checks without network downloads, `ablate` runs all pipelines on a
14-passage / 15-query distinct-topic sanity fixture (`k = 3`). Because topics are non-overlapping,
all four lexical variants (`bm25`, `dense` TF-IDF cosine, `hybrid` RRF, and `reranked`) converge
to `recall@3 = 1.000`, `MRR = 0.967`, `nDCG@3 = 0.975` - verifying metric and pipeline wiring
rather than discriminating retriever quality.
