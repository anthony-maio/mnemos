from datetime import datetime, timedelta, timezone

import pytest

from mnemos.jobs import DurableJob
from mnemos.maintenance import MaintenanceScheduler
from mnemos.utils.storage import SQLiteStore


@pytest.mark.asyncio
async def test_idle_and_starvation_deadlines_use_durable_jobs(tmp_path):
    store = SQLiteStore(str(tmp_path / "scheduler.sqlite"))
    start = datetime.now(timezone.utc)
    clock = [start]
    job = DurableJob(kind="test", created_at=start, available_at=start)
    store.enqueue_job(job)
    observed = []

    async def process(claimed):
        observed.append(claimed.id)
        assert store.complete_job(
            claimed.id, owner=claimed.lease_owner, lease_token=claimed.lease_token, now=clock[0]
        )

    scheduler = MaintenanceScheduler(store, process, clock=lambda: clock[0])
    try:
        assert await scheduler.run_once() == 0
        clock[0] += timedelta(seconds=59)
        assert await scheduler.run_once() == 0
        clock[0] += timedelta(seconds=1)
        assert await scheduler.run_once() == 1
        assert observed == [job.id]
        old_job = DurableJob(kind="test", created_at=start, available_at=start)
        store.enqueue_job(old_job)
        clock[0] += timedelta(seconds=241)
        scheduler.note_activity()
        assert await scheduler.run_once() == 1
        assert observed == [job.id, old_job.id]
    finally:
        store.close()


@pytest.mark.asyncio
async def test_processor_failure_remains_durable_and_visible(tmp_path):
    path = str(tmp_path / "failure.sqlite")
    store = SQLiteStore(path)
    store.enqueue_job(DurableJob(kind="extract"))

    async def unavailable(job):
        raise ConnectionError("model offline")

    scheduler = MaintenanceScheduler(store, unavailable)
    assert await scheduler.run_once(force=True) == 0
    assert scheduler.health()["last_error"] == "ConnectionError"
    store.close()
    reopened = SQLiteStore(path)
    try:
        job = reopened.list_jobs()[0]
        assert job.status == "pending"
        assert job.attempt_count == 1
        assert job.last_error == "ConnectionError"
    finally:
        reopened.close()
