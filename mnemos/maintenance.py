"""Bounded background maintenance with durable leases and observable failures."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from .jobs import DurableJob
from .utils.storage import (
    ConflictCanonicalCollisionError,
    CorrectionCanonicalCollisionError,
    MemoryStore,
)


class JobLeaseLostError(RuntimeError):
    """Active processing no longer owns a renewable job lease."""


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
        lease_seconds: int = 300,
        lease_renewal_seconds: float | None = None,
        processor_cancel_seconds: float = 5,
        processed_evidence_days: int = 30,
        after_batch: Callable[[], object] | None = None,
        on_error: Callable[[BaseException], object] | None = None,
    ) -> None:
        if interval_seconds <= 0 or idle_seconds < 0 or max_queue_age_seconds <= 0:
            raise ValueError("Invalid maintenance schedule")
        if not 1 <= batch_limit <= 100:
            raise ValueError("Maintenance batches must contain 1 to 100 jobs")
        if processed_evidence_days < 0:
            raise ValueError("Evidence retention days cannot be negative")
        renewal_seconds = (
            lease_seconds / 3 if lease_renewal_seconds is None else lease_renewal_seconds
        )
        if not math.isfinite(lease_seconds) or not 0 < renewal_seconds < lease_seconds:
            raise ValueError("Lease renewal interval must be positive and shorter than the lease")
        if not math.isfinite(processor_cancel_seconds) or processor_cancel_seconds <= 0:
            raise ValueError("Processor cancellation grace must be finite and positive")
        self.processor_cancel_seconds = processor_cancel_seconds
        self.lease_seconds = lease_seconds
        self.lease_renewal_seconds = renewal_seconds
        self.store = store
        self.processor = processor
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.interval_seconds = interval_seconds
        self.idle_seconds = idle_seconds
        self.max_queue_age_seconds = max_queue_age_seconds
        self.batch_limit = batch_limit
        self.processed_evidence_days = processed_evidence_days
        self.after_batch = after_batch
        self.on_error = on_error
        self.owner = str(uuid4())
        self.last_activity = self.clock()
        self.last_success: datetime | None = None
        self.last_error: str | None = None
        self.processed_count = 0
        self.failed_count = 0
        self._running = False
        self._scheduler_running = False
        self._stop = asyncio.Event()

    def _notify_error(self, error: BaseException) -> None:
        if self.on_error is None:
            return
        try:
            self.on_error(error)
        except Exception:
            # Observability must not alter maintenance's durable error path.
            pass

    def note_activity(self) -> None:
        self.last_activity = self.clock()

    async def _process_with_lease(self, job: DurableJob) -> None:
        task = asyncio.ensure_future(self.processor(job))
        # A processor that ignores cancellation can outlive this scheduler call.
        # Consume any eventual exception even after its token has been revoked.
        task.add_done_callback(
            lambda finished: None if finished.cancelled() else finished.exception()
        )
        try:
            while True:
                done, _ = await asyncio.wait((task,), timeout=self.lease_renewal_seconds)
                if done:
                    await task
                    return
                if not self.store.renew_job_lease(
                    job.id,
                    owner=self.owner,
                    lease_token=job.lease_token,
                    lease_seconds=self.lease_seconds,
                    now=self.clock(),
                ):
                    raise JobLeaseLostError("Maintenance job lease could not be renewed")
        finally:
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait((task,), timeout=self.processor_cancel_seconds)
                finally:
                    if not task.done():
                        # Bound shutdown even for a non-cooperative processor. Revoke
                        # its token before allowing any eventual result to arrive.
                        try:
                            self.store.fail_job(
                                job.id,
                                owner=self.owner,
                                lease_token=job.lease_token,
                                error="processor_cancellation_timeout",
                                now=self.clock(),
                            )
                        finally:
                            task.cancel()

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
                jobs = self.store.claim_jobs(
                    owner=self.owner, limit=1, lease_seconds=self.lease_seconds, now=self.clock()
                )
                if not jobs:
                    break
                job = jobs[0]
                try:
                    # The processor must atomically commit results and acknowledge the job.
                    await self._process_with_lease(job)
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
                    self.last_error = (
                        "conflict_canonical_collision"
                        if isinstance(exc, ConflictCanonicalCollisionError)
                        else (
                            "correction_canonical_collision"
                            if isinstance(exc, CorrectionCanonicalCollisionError)
                            else type(exc).__name__
                        )
                    )
                    self.failed_count += 1
                    self._notify_error(exc)
                    if not isinstance(exc, JobLeaseLostError):
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
                    self._notify_error(exc)
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
