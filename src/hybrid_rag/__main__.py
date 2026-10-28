"""Enable ``python -m hybrid_rag`` to invoke the command-line interface."""

from __future__ import annotations

from hybrid_rag.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
