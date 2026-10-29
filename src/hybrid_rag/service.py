"""A dependency-free HTTP service exposing hybrid retrieval.

The service indexes the bundled corpus with every pipeline once at startup, then
answers retrieval requests against a chosen pipeline. A pure handler is separated
from the socket binding for testing.

Endpoints
---------
``GET  /health``     liveness, pipeline names, and chunk count.
``POST /retrieve``   body ``{"query": str, "pipeline"?: str, "k"?: int}``;
                     returns the ranked chunks with scores.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from hybrid_rag.core.chunking import FixedTokenChunker, chunk_documents
from hybrid_rag.core.retriever import Retriever
from hybrid_rag.data.sample import corpus
from hybrid_rag.pipelines import available_pipelines, build_pipeline

JsonObject = dict[str, Any]
_MAX_QUERY_CHARS = 2000
_MAX_K = 50
_CHUNKER = FixedTokenChunker(chunk_size=200, overlap=0)


@dataclass(frozen=True)
class RetrievalService:
    """Holds one indexed retriever per pipeline and answers requests."""

    retrievers: dict[str, Retriever]
    chunk_count: int

    @classmethod
    def from_corpus(cls) -> RetrievalService:
        chunks = chunk_documents(corpus(), _CHUNKER)
        retrievers: dict[str, Retriever] = {}
        for name in available_pipelines():
            retriever = build_pipeline(name)
            retriever.index(chunks)
            retrievers[name] = retriever
        return cls(retrievers=retrievers, chunk_count=len(chunks))

    def retrieve(self, query: str, pipeline: str, k: int) -> JsonObject:
        results = self.retrievers[pipeline].retrieve(query, k=k)
        return {
            "query": query,
            "pipeline": pipeline,
            "results": [
                {"id": scored.id, "score": round(scored.score, 6), "text": scored.chunk.text}
                for scored in results
            ],
        }


def _validate_retrieve(
    body: Mapping[str, Any], pipelines: Mapping[str, Retriever]
) -> tuple[str, str, int]:
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("'query' must be a non-empty string")
    if len(query) > _MAX_QUERY_CHARS:
        raise ValueError(f"'query' exceeds {_MAX_QUERY_CHARS} characters")
    pipeline = body.get("pipeline", "hybrid")
    if pipeline not in pipelines:
        raise ValueError(f"'pipeline' must be one of: {', '.join(pipelines)}")
    k = body.get("k", 5)
    if not isinstance(k, int) or not 1 <= k <= _MAX_K:
        raise ValueError(f"'k' must be an integer in [1, {_MAX_K}]")
    return query, pipeline, k


def handle(
    method: str, path: str, body: Mapping[str, Any], service: RetrievalService
) -> tuple[int, JsonObject]:
    """Pure request handler: route a request to a status code and payload."""

    if method == "GET" and path == "/health":
        return 200, {
            "status": "ok",
            "pipelines": list(service.retrievers),
            "chunks": service.chunk_count,
        }
    if method == "POST" and path == "/retrieve":
        try:
            query, pipeline, k = _validate_retrieve(body, service.retrievers)
        except ValueError as error:
            return 400, {"error": str(error)}
        return 200, service.retrieve(query, pipeline, k)
    return 404, {"error": f"no route for {method} {path}"}


class _Handler(BaseHTTPRequestHandler):  # pragma: no cover - exercised over sockets
    service: RetrievalService

    def _respond(self, status: int, payload: JsonObject) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        status, payload = handle("GET", self.path, {}, self.service)
        self._respond(status, payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._respond(400, {"error": "request body is not valid JSON"})
            return
        status, payload = handle("POST", self.path, body, self.service)
        self._respond(status, payload)

    def log_message(self, *_: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    """Start the HTTP service (blocking)."""

    handler = type("_BoundHandler", (_Handler,), {"service": RetrievalService.from_corpus()})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"hybrid-rag service listening on http://{host}:{port}")
    server.serve_forever()
