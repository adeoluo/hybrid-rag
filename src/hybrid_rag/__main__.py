"""Entrypoint: `python -m hybrid_rag`."""

from __future__ import annotations

from hybrid_rag import __version__
from hybrid_rag.configuration import settings


def main() -> None:
    header = f"Hybrid Rag v{__version__}"
    status = f"environment={settings.environment}, seed={settings.seed}"
    print(f"{header} - {status}")


if __name__ == "__main__":
    main()
