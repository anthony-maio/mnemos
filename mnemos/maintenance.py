"""Bounded background maintenance with durable leases and observable failures."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from .jobs import DurableJob
from .utils.storage import MemoryStore


class MaintenanceScheduler:
    def __init__(
        self,
        store: MemoryStore,
        processor: Callable[[DurableJob], Awaitable[None]],
        *,
        clock: Callable[[], datetime] | None = None,
        interval_seconds: float = 30,
        idle_seconds: float = 60,
        max_queue_age_seconds: float = 300,
        batch_limit: int = 100,
        processed_evidence_days: int = 30,
        after_batch: Callable[[], object] | None = None,
    ) -> None:
        if interval_seconds <= 0 or idle_seconds < 0 or max_queue_age_seconds <= 0:
            raise ValueError("Invalid maintenance schedule")
        if not 1 <= batch_limit <= 100:
            raise ValueError("Maintenance batches must contain 1 to 100 jobs")
        if processed_evidence_days < 0:
            raise ValueError("Evidence retention days cannot be negative")
        self.store = store
        self.processor = processor
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.interval_seconds = interval_seconds
        self.idle_seconds = idle_seconds
        self.max_queue_age_seconds = max_queue_age_seconds
        self.batch_limit = batch_limit
        self.processed_evidence_days = processed_evidence_days
        self.after_batch = after_batch
        self.owner = str(uuid4())
        self.last_activity = self.clock()
        self.last_success: datetime | None = None
        self.last_error: str | None = None
        self.processed_count = 0
        self.failed_count = 0
        self._running = False
        self._scheduler_running = False
        self._stop = asyncio.Event()

    def note_activity(self) -> None:
        self.last_activity = self.clock()

    async def run_once(self, *, force: bool = False) -> int:
        if self._running:
            return 0
        now = self.clock()
        summary = self.store.job_summary(now=now)
        overdue = (
            summary["pending"] > 0
            and summary["oldest_pending_seconds"] >= self.max_queue_age_seconds
        )
        if (
            not force
            and not overdue
            and (now - self.last_activity).total_seconds() < self.idle_seconds
        ):
            return 0
        self._running = True
        completed = 0
        try:
            for _ in range(self.batch_limit):
                jobs = self.store.claim_jobs(owner=self.owner, limit=1, now=self.clock())
                if not jobs:
                    break
                job = jobs[0]
                try:
                    # The processor must atomically commit results and acknowledge the job.
                    await self.processor(job)
                    current = self.store.get_job(job.id)
                    if current is None or current.status != "completed":
                        raise RuntimeError("Processor returned without committing the leased job")
                    self.last_success = self.clock()
                    self.last_error = None
                    self.processed_count += 1
                    completed += 1
                except asyncio.CancelledError:
                    # Shutdown leaves the lease recoverable after expiry.
                    raise
                except Exception as exc:
                    self.last_error = type(exc).__name__
                    self.failed_count += 1
                    self.store.fail_job(
                        job.id,
                        owner=self.owner,
                        lease_token=job.lease_token,
                        error=self.last_error,
                        now=self.clock(),
                    )
                await asyncio.sleep(0)
            prune = getattr(self.store, "prune_processed_evidence", None)
            if callable(prune) and self.processed_evidence_days > 0:
                prune(
                    before=self.clock() - timedelta(days=self.processed_evidence_days),
                    limit=self.batch_limit,
                )
            if self.after_batch is not None:
                self.after_batch()
        finally:
            self._running = False
        return completed

    async def run(self) -> None:
        self._scheduler_running = True
        try:
            while not self._stop.is_set():
                try:
                    await self.run_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.last_error = type(exc).__name__
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
                except asyncio.TimeoutError:
                    pass
        finally:
            self._scheduler_running = False

    def stop(self) -> None:
        self._stop.set()

    def health(self) -> dict[str, object]:
        now = self.clock()
        return {
            **self.store.job_summary(now=now),
            "running": self._running,
            "scheduler_running": self._scheduler_running,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_error": self.last_error,
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
        }
