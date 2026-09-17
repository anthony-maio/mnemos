"""
mnemos/observability.py — Structured JSON logging helpers.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
import threading
from typing import Any

SERVICE_LOG_MAX_BYTES = 1_048_576
SERVICE_LOG_BACKUPS = 2

_SERVICE_LOG_FIELDS: dict[str, frozenset[str]] = {
    "mnemos.startup": frozenset({"timestamp", "event", "status", "backend", "provider"}),
    "mnemos.provider_failure": frozenset(
        {"timestamp", "event", "provider", "operation", "error_type"}
    ),
    "mnemos.retrieval_latency": frozenset(
        {
            "timestamp",
            "event",
            "latency_ms",
            "result_count",
            "top_k",
            "scope",
            "allowed_scopes",
            "store_backend",
        }
    ),
}

_SERVICE_LOG_LOCK = threading.Lock()
_ACTIVE_SERVICE_LOG_BINDING: ServiceLogBinding | None = None


class _PrivateRotatingFileHandler(RotatingFileHandler):
    def __init__(self, *args: Any, protect_path: Any, **kwargs: Any) -> None:
        self._protect_path = protect_path
        super().__init__(*args, **kwargs)

    def _open(self) -> Any:
        stream: Any
        if os.name == "nt":
            stream = super()._open()
        else:
            flags = os.O_WRONLY | os.O_CREAT
            if "a" in self.mode:
                flags |= os.O_APPEND
            elif "w" in self.mode:
                flags |= os.O_TRUNC
            elif "x" in self.mode:
                flags |= os.O_EXCL
            fd = os.open(self.baseFilename, flags, 0o600)
            try:
                stream = os.fdopen(
                    fd,
                    self.mode,
                    encoding=self.encoding,
                    errors=self.errors,
                )
            except BaseException:
                os.close(fd)
                raise
        try:
            self._protect_path(Path(self.baseFilename))
        except BaseException:
            stream.close()
            raise
        return stream


class ServiceLogBinding:
    """The service-owned handler and logger state needed for cleanup."""

    def __init__(self, logger: logging.Logger, handler: logging.Handler, prior_level: int) -> None:
        self.logger = logger
        self.handler = handler
        self.prior_level = prior_level


class _ServiceLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            payload = json.loads(record.getMessage())
        except (TypeError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        event = payload.get("event")
        if not isinstance(event, str):
            return False
        allowed = _SERVICE_LOG_FIELDS.get(event)
        if allowed is None:
            return False
        if event == "mnemos.provider_failure" and "error_type" not in payload:
            return False
        return True


class _ServiceJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        try:
            payload = json.loads(record.getMessage())
        except (TypeError, json.JSONDecodeError):
            return ""
        if not isinstance(payload, dict):
            return ""
        event = payload.get("event")
        allowed = (
            _SERVICE_LOG_FIELDS.get(event, frozenset()) if isinstance(event, str) else frozenset()
        )
        return json.dumps(
            {key: payload[key] for key in allowed if key in payload},
            sort_keys=True,
            ensure_ascii=True,
        )


def install_service_file_logging(path: str | Path, *, protect_path: Any) -> ServiceLogBinding:
    """Attach one private, rotating JSON handler to the Mnemos logger."""
    global _ACTIVE_SERVICE_LOG_BINDING
    target = Path(path).expanduser()
    with _SERVICE_LOG_LOCK:
        if _ACTIVE_SERVICE_LOG_BINDING is not None:
            raise RuntimeError("a Mnemos service log is already active in this process")
        target.parent.mkdir(parents=True, exist_ok=True)
        protect_path(target.parent)
        logger = logging.getLogger("mnemos")
        prior_level = logger.level
        handler = _PrivateRotatingFileHandler(
            target,
            maxBytes=SERVICE_LOG_MAX_BYTES,
            backupCount=SERVICE_LOG_BACKUPS,
            encoding="utf-8",
            delay=False,
            protect_path=protect_path,
        )
        handler.setFormatter(_ServiceJsonFormatter())
        handler.addFilter(_ServiceLogFilter())
        try:
            logger.addHandler(handler)
            if logger.level == logging.NOTSET or logger.getEffectiveLevel() > logging.INFO:
                logger.setLevel(logging.INFO)
            binding = ServiceLogBinding(logger, handler, prior_level)
            _ACTIVE_SERVICE_LOG_BINDING = binding
            return binding
        except BaseException:
            handler.close()
            raise


def remove_service_file_logging(binding: ServiceLogBinding | None) -> None:
    """Detach only the handler installed by ``install_service_file_logging``."""
    global _ACTIVE_SERVICE_LOG_BINDING
    if binding is None:
        return
    with _SERVICE_LOG_LOCK:
        binding.logger.removeHandler(binding.handler)
        binding.handler.close()
        if _ACTIVE_SERVICE_LOG_BINDING is binding:
            binding.logger.setLevel(binding.prior_level)
            _ACTIVE_SERVICE_LOG_BINDING = None


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
