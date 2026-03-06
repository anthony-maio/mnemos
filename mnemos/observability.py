"""
mnemos/observability.py — Structured JSON logging helpers.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any


def configure_logging() -> None:
    """Configure process logging once, defaulting to JSON-friendly format."""
    root = logging.getLogger()
    if root.handlers:
        return

    level_name = os.getenv("MNEMOS_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")


def log_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Emit a structured JSON log event."""
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
    }
    payload.update(fields)
    logging.getLogger("mnemos").log(level, json.dumps(payload, default=str))
