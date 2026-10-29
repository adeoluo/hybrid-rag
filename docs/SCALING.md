# Scaling and Design Tradeoffs - Hybrid RAG

How the reference design grows, and the tradeoffs taken.

## Reference architecture (today)

A single process indexes the corpus with every pipeline in memory and serves retrieval over
a standard-library HTTP service. Shared-nothing and stateless per request.

## Scaling path

| Stage | Bottleneck | Change |
|---|---|---|
| One process | CPU, global interpreter lock | One replica per core |
| One host | Index memory times pipelines | Horizontal replicas; index fewer pipelines per replica |
| Large corpus | Index memory | Shard the corpus; scatter/gather across shards |
| Heavy pipelines | Rerank/diversify latency | Route heavy pipelines to a dedicated pool |
| Hot queries | Repeated identical work | Read-through cache keyed by (query, pipeline, k) |

Because replicas share nothing, throughput is approximately linear in replica count.

## Substantial tradeoffs

- **In-memory indexes vs an external store.** In memory gives millisecond retrieval and zero
  dependencies, bounded by corpus size and the number of pipelines per replica. An external
  vector or inverted-index store lifts the ceiling but adds a network hop and a dependency.
  Retrievers sit behind a protocol so the store can be swapped.
- **Many pipelines vs one.** Offering six pipelines exposes the latency-for-quality tradeoff
  explicitly, at the cost of more index memory and a selection decision. A single fixed
  pipeline is simpler but hides the tradeoff.
- **Fusion and reranking vs raw retrieval.** Reciprocal-rank fusion, cross-encoder-style
  reranking, and maximal marginal relevance improve quality or diversity but each adds a
  stage and latency (measured). The pipeline makes the cost explicit and opt-in.
- **Threads vs processes.** Retrieval is pure Python and GIL-bound, so processes, not
  threads, are the unit of scaling. The service is stateless to make that trivial.

## What would change under real load

Shard the corpus, run process-per-core replicas, route heavy pipelines to their own pool,
and add a per-(query, pipeline, k) cache. The retrievers and fusion are unchanged; only what
sits behind their interfaces changes.
