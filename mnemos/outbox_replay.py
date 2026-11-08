"""Explicitly owned replay lifecycle for an opt-in resident capture worker."""

from __future__ import annotations

import asyncio
import math
from typing import Any

from .client import MnemosClient


class OutboxReplayPump:
    """Drain through a dedicated client; disk remains authoritative on shutdown.

    The owner must start and close this pump on its event loop. Do not share its
    client with concurrent hook handling. No process or runtime is started by
    constructing the pump. The supplied client's autostart policy governs replay.
    """

    def __init__(self, client: MnemosClient, *, retry_seconds: float = 5.0) -> None:
        if not math.isfinite(retry_seconds) or retry_seconds <= 0:
            raise ValueError("Replay interval must be positive and finite")
        self.client = client
        self.retry_seconds = retry_seconds
        self.last_result: dict[str, Any] | None = None
        self.last_error_type: str | None = None
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("Replay pump is closed")
        if self._task is None:
            self._wake.set()  # Includes events left by an earlier process.
            self._task = asyncio.create_task(self._run())

    def notify(self) -> None:
        if self._closed or self._task is None:
            raise RuntimeError("Replay pump is not running")
        self._wake.set()

    async def _run(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.retry_seconds)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            try:
                self.client.begin_host_event()
                result = await self.client.replay_outbox()
            except Exception as exc:
                self.last_result = None
                self.last_error_type = type(exc).__name__
            else:
                self.last_result = result
                self.last_error_type = None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.client.close()
