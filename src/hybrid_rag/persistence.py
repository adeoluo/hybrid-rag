"""Persisting a chunked corpus to disk.

Chunking a large corpus is work worth doing once. :class:`ChunkStore` serializes a list of
chunks to JSON and reloads them, so an index can be rebuilt from saved chunks without
re-reading and re-chunking the source documents. Retrievers index these chunks exactly as
they would freshly produced ones.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from hybrid_rag.core.types import Chunk


class ChunkStore:
    """Serializes and reloads a list of chunks."""

    @staticmethod
    def save(chunks: Sequence[Chunk], path: Path) -> None:
        """Write ``chunks`` to ``path`` as JSON."""
        payload = [
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> list[Chunk]:
        """Read chunks previously written by :meth:`save`."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [
            Chunk(
                id=item["id"],
                document_id=item["document_id"],
                text=item["text"],
                metadata=dict(item.get("metadata", {})),
            )
            for item in payload
        ]
