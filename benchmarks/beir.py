"""Retrieval quality and speed on a BEIR dataset (default: SciFact).

BEIR (Thakur et al., 2021) is the standard zero-shot retrieval benchmark. Each dataset
ships a corpus, test queries, and graded relevance judgments, so the numbers here are
directly comparable with published ones. Download a dataset, for example
https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip, unzip it,
and point ``--data`` at the folder.

Metrics follow BEIR: nDCG@10 with graded gains (pytrec_eval's definition) and capped
recall@100, which divides by min(100, relevant documents) so that queries with hundreds of
relevant documents (as in TREC-COVID) can still reach 1.0. On SciFact it equals plain recall.
With ``--rank-bm25`` the script also scores the ``rank_bm25`` package's BM25Okapi on the
same tokens, as an independent BM25 implementation.

    uv run --with rank_bm25 python benchmarks/beir.py --data /path/to/scifact --rank-bm25
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from hybrid_rag.core.tokenization import tokenize
from hybrid_rag.core.types import Chunk
from hybrid_rag.pipelines import build_pipeline

DEFAULT_PIPELINES = ("bm25", "tfidf", "hybrid", "reranked", "hybrid-all", "mmr")


def load(data: Path, split: str) -> tuple[list[Chunk], dict[str, str], dict[str, dict[str, int]]]:
    chunks = []
    with (data / "corpus.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            text = f"{row.get('title', '')} {row['text']}".strip()
            if text:
                chunks.append(Chunk(id=row["_id"], document_id=row["_id"], text=text))
    qrels: dict[str, dict[str, int]] = {}
    with (data / "qrels" / f"{split}.tsv").open() as f:
        next(f)
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    queries = {}
    with (data / "queries.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in qrels:
                queries[row["_id"]] = row["text"]
    return chunks, queries, qrels


def ndcg_at(ranked: list[str], rels: dict[str, int], k: int) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at(ranked: list[str], rels: dict[str, int], k: int) -> float:
    return len(set(ranked[:k]) & set(rels)) / min(k, len(rels))


def evaluate(run: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    n = len(qrels)
    return {
        "ndcg@10": sum(ndcg_at(run.get(q, []), r, 10) for q, r in qrels.items()) / n,
        "recall@100": sum(recall_at(run.get(q, []), r, 100) for q, r in qrels.items()) / n,
    }


def run_rank_bm25(chunks: list[Chunk], queries: dict[str, str]) -> tuple[dict, float, float]:
    from rank_bm25 import BM25Okapi

    start = time.perf_counter()
    bm25 = BM25Okapi([tokenize(c.text) for c in chunks], k1=1.5, b=0.75)
    build = time.perf_counter() - start
    ids = [c.id for c in chunks]
    run = {}
    start = time.perf_counter()
    for qid, text in queries.items():
        scores = bm25.get_scores(tokenize(text))
        order = sorted(range(len(ids)), key=lambda i: -scores[i])[:100]
        run[qid] = [ids[i] for i in order]
    return run, build, (time.perf_counter() - start) / len(queries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--pipelines", default=",".join(DEFAULT_PIPELINES))
    parser.add_argument("--rank-bm25", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    chunks, queries, qrels = load(args.data, args.split)
    print(f"{args.data.name}: {len(chunks):,} documents, {len(queries)} queries")
    print("| Retriever | nDCG@10 | Recall@100 | Index build | Query |")
    print("|---|---:|---:|---:|---:|")
    rows = []
    for name in args.pipelines.split(","):
        retriever = build_pipeline(name)
        start = time.perf_counter()
        retriever.index(chunks)
        build = time.perf_counter() - start
        run = {}
        start = time.perf_counter()
        for qid, text in queries.items():
            run[qid] = [s.chunk.id for s in retriever.retrieve(text, k=100)]
        per_query = (time.perf_counter() - start) / len(queries)
        rows.append(
            {
                "retriever": name,
                **evaluate(run, qrels),
                "build_s": build,
                "query_ms": per_query * 1000,
            }
        )
    if args.rank_bm25:
        run, build, per_query = run_rank_bm25(chunks, queries)
        rows.append(
            {
                "retriever": "rank_bm25 BM25Okapi",
                **evaluate(run, qrels),
                "build_s": build,
                "query_ms": per_query * 1000,
            }
        )
    for r in rows:
        print(
            f"| {r['retriever']} | {r['ndcg@10']:.3f} | {r['recall@100']:.3f} | "
            f"{r['build_s']:.2f} s | {r['query_ms']:.2f} ms |"
        )
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "dataset": args.data.name,
                    "documents": len(chunks),
                    "queries": len(queries),
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
