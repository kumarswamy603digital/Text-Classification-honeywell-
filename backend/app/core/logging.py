"""Structured logging setup (stdlib, dependency-light)."""
from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    # Keep third-party noise down
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
