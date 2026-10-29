# Operations Guide - Hybrid RAG

The operational contract of the retrieval service.

## Running the service

```bash
uv run python -m hybrid_rag serve --host 0.0.0.0 --port 8000
```

A standard-library HTTP server with **zero runtime dependencies**. Every pipeline is
indexed over the corpus once at startup, then the service serves stateless requests.

## Endpoints

| Method | Path | Body | Success |
|---|---|---|---|
| GET | `/health` | none | status, pipeline names, chunk count |
| GET | `/ready` | none | readiness once every pipeline is indexed |
| GET | `/metrics` | none | per-route request, error, and latency metrics (Prometheus text) |
| POST | `/retrieve` | `{"query", "pipeline"?, "k"?}` | ranked chunks with scores |

`pipeline` defaults to `hybrid` and must be one of the advertised names; `k` is bounded.
Errors return JSON `{"error": ...}` with a 4xx status.

## Service level objectives

Measured by `benchmarks/stress.py` over a 300 document synthetic corpus at four workers:

| Pipeline | Throughput | p50 | p99 |
|---|---:|---:|---:|
| dense | about 3600/s | 0.3 ms | 19 ms |
| bm25 | about 2600/s | 0.4 ms | 19 ms |
| hybrid | about 1500/s | 0.6 ms | 32 ms |
| reranked | about 900/s | 1.1 ms | 39 ms |
| hybrid-all | about 500/s | 2.0 ms | 46 ms |
| mmr | about 470/s | 1.3 ms | 83 ms |

The spread is the point: more retrieval stages cost latency, so the pipeline is an explicit
latency-for-quality dial, selectable per request.

## Operational constraints

- **Retrieval is CPU bound and holds the global interpreter lock.** Throughput is flat
  across threads in one process and the p99 tail grows with concurrency. Scale with
  **processes**, not threads.
- **Indexes are in memory and immutable after startup.** Memory grows with corpus size and
  with the number of pipelines indexed; updates require a restart or blue/green swap.
- **Requests are bounded**: queries to 2000 characters and `k` to 50.
- **No persistence, no authentication.** Front the service with a gateway.

## Failure modes and responses

| Symptom | Likely cause | Response |
|---|---|---|
| High p99 on `mmr`/`hybrid-all` | Extra rerank/diversify stages | Choose a lighter pipeline per request |
| Rising p99 under load | Concurrency above one worker per core | Add replicas; cap concurrency |
| Memory growth at startup | Large corpus times many pipelines | Index fewer pipelines per replica; shard corpus |

## Capacity planning

Throughput depends on the pipeline. Size replicas for the slowest pipeline a client may
request, or route heavy pipelines to a dedicated pool.
