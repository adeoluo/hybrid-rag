# End-to-End Case Study - Choosing a Retrieval Pipeline

How the same query is served by six pipelines, and how to pick one under a latency budget.

## Scenario

A product must retrieve relevant passages for a query. The team can afford a latency budget
and wants the best quality within it. Hybrid RAG exposes six pipelines, from a single
lexical retriever to a hybrid-plus-rerank-plus-diversify stack.

## Serving

```bash
curl -s -X POST localhost:8000/retrieve \
  -d '{"query": "neural networks", "pipeline": "hybrid", "k": 5}'
```

The response ranks chunks with scores. Switching `pipeline` changes the quality and the
latency of the same request.

## Quality vs latency

On the labeled queries (`python -m hybrid_rag ablate`) the pipelines are close on recall and
nDCG, because the corpus is clean; the differences show up under load. From
`benchmarks/stress.py`:

| Pipeline | Throughput | p99 latency | When to use |
|---|---:|---:|---|
| dense / bm25 | about 3000/s | about 19 ms | Tight latency budgets, high traffic |
| hybrid | about 1500/s | about 32 ms | Balanced default |
| reranked | about 900/s | about 39 ms | Quality-sensitive, moderate traffic |
| hybrid-all / mmr | about 500/s | about 46-83 ms | Diversity or maximum quality, low traffic |

The right choice is a per-request decision: the service lets a caller trade latency for
quality explicitly by naming the pipeline.

## What an operator sees

- **A latency dial.** Pipeline selection is a first-class, per-request knob.
- **Observability.** Each result carries a score, so callers can threshold or re-rank.
- **Predictable cost.** `k` and query length are bounded, so a request cannot run away.

## Takeaway

Owning the full retrieval stack from scratch makes the latency-for-quality tradeoff explicit
and selectable per request, rather than hidden inside a managed black box.
