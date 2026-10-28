"""Command-line interface for the hybrid retrieval system.

Subcommands:

- ``query`` - index the bundled corpus with a chosen pipeline and show the top results.
- ``ablate`` - evaluate every pipeline on the bundled labeled queries and print a table,
  the headline result of the project.

The CLI wires together chunking, the pipelines, the bundled data, and the evaluation
harness; it contains no retrieval logic of its own.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from hybrid_rag import __version__
from hybrid_rag.core.chunking import FixedTokenChunker, chunk_documents
from hybrid_rag.data.sample import corpus, labeled_queries
from hybrid_rag.evaluation.harness import evaluate_retriever
from hybrid_rag.pipelines import available_pipelines, build_pipeline

# The sample passages are short, so one chunk per document keeps ids predictable.
_CHUNKER = FixedTokenChunker(chunk_size=200, overlap=0)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="hybrid-rag",
        description="A from-scratch hybrid retrieval system with evaluation.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query", help="Retrieve over the bundled corpus.")
    query_parser.add_argument("text", help="The query text.")
    query_parser.add_argument(
        "--pipeline", choices=available_pipelines(), default="hybrid", help="Retriever."
    )
    query_parser.add_argument("--k", type=int, default=3, help="Number of results.")
    query_parser.set_defaults(handler=_handle_query)

    ablate_parser = subparsers.add_parser(
        "ablate", help="Evaluate every pipeline on the bundled queries."
    )
    ablate_parser.add_argument("--k", type=int, default=5, help="Evaluation cutoff.")
    ablate_parser.set_defaults(handler=_handle_ablate)
    return parser


def _handle_query(args: argparse.Namespace) -> int:
    chunks = chunk_documents(corpus(), _CHUNKER)
    retriever = build_pipeline(args.pipeline)
    retriever.index(chunks)
    results = retriever.retrieve(args.text, k=args.k)
    print(f"Top {len(results)} results for {args.text!r} ({args.pipeline}):")
    for rank, scored in enumerate(results, start=1):
        snippet = scored.chunk.text[:80].replace("\n", " ")
        print(f"  {rank}. [{scored.score:.4f}] {scored.chunk.id}: {snippet}...")
    return 0


def _handle_ablate(args: argparse.Namespace) -> int:
    chunks = chunk_documents(corpus(), _CHUNKER)
    queries = labeled_queries()
    print(f"Ablation on {len(queries)} queries at k={args.k}:\n")
    header = f"{'pipeline':<10}  recall  precision   MRR   nDCG"
    print(header)
    print("-" * len(header))
    for name in available_pipelines():
        retriever = build_pipeline(name)
        retriever.index(chunks)
        report = evaluate_retriever(retriever, queries, k=args.k)
        print(
            f"{name:<10}  {report.recall:.3f}   {report.precision:.3f}    "
            f"{report.mrr:.3f}  {report.ndcg:.3f}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    exit_code: int = handler(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
