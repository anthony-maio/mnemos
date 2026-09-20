import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from mnemos.jobs import DurableJob
from mnemos.maintenance import MaintenanceScheduler
from mnemos.utils.storage import SQLiteStore


def test_renewal_preserves_fences_and_bounds_crash_recovery(tmp_path):
    store = SQLiteStore(str(tmp_path / "leases.sqlite"))
    start = datetime.now(timezone.utc)
    job = DurableJob(kind="test", available_at=start)
    store.enqueue_job(job)
    first = store.claim_jobs(owner="a", now=start)[0]

    def renew(owner="a", token=first.lease_token, seconds=300, elapsed=100):
        return store.renew_job_lease(
            job.id,
            owner=owner,
            lease_token=token,
            lease_seconds=seconds,
            now=start + timedelta(seconds=elapsed),
        )

    try:
        before = store.get_job(job.id)
        assert not renew(owner="b")
        assert not renew(token="stale")
        assert not renew(token=None)
        assert store.get_job(job.id) == before
        assert renew()
        assert store.get_job(job.id).lease_until == start + timedelta(seconds=400)
        assert renew(seconds=10)
        assert store.get_job(job.id).lease_until == start + timedelta(seconds=400)
        assert store.get_job(job.id).attempt_count == 1
        assert not renew(elapsed=400)
        second = store.claim_jobs(owner="a", now=start + timedelta(seconds=400))[0]
        assert not renew(elapsed=401)
        assert not store.complete_job(
            job.id, owner="a", lease_token=first.lease_token, now=start + timedelta(seconds=401)
        )
        assert store.complete_job(
            job.id, owner="a", lease_token=second.lease_token, now=start + timedelta(seconds=401)
        )
        assert not renew(token=second.lease_token, elapsed=402)
        assert store.get_job(job.id).status == "completed"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_long_processing_renews_then_commits_without_reclaim(tmp_path, monkeypatch):
    store = SQLiteStore(str(tmp_path / "long.sqlite"))
    start = datetime.now(timezone.utc)
    clock = [start]
    store.enqueue_job(DurableJob(kind="test", available_at=start))
    renewed = asyncio.Event()
    original = store.renew_job_lease
    renewals = []

    def renew(*args, **kwargs):
        result = original(*args, **kwargs)
        renewals.append(result)
        renewed.set()
        return result

    monkeypatch.setattr(store, "renew_job_lease", renew)

    async def process(job):
        for elapsed in (200, 400, 600):
            renewed.clear()
            clock[0] = start + timedelta(seconds=elapsed)
            await renewed.wait()
            assert store.claim_jobs(owner="competitor", now=clock[0]) == []
        assert store.complete_job(
            job.id, owner=job.lease_owner, lease_token=job.lease_token, now=clock[0]
        )

    scheduler = MaintenanceScheduler(
        store, process, clock=lambda: clock[0], lease_renewal_seconds=0.001, batch_limit=1
    )
    try:
        assert await asyncio.wait_for(scheduler.run_once(force=True), 5) == 1
        assert renewals == [True, True, True]
        assert store.list_jobs()[0].attempt_count == 1
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("lose_ownership", [False, True])
async def test_shutdown_or_lost_lease_cancels_processor(tmp_path, lose_ownership):
    store = SQLiteStore(str(tmp_path / "cancel.sqlite"))
    start = datetime.now(timezone.utc)
    clock = [start]
    store.enqueue_job(DurableJob(kind="test", available_at=start))
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def process(job):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    scheduler = MaintenanceScheduler(
        store, process, clock=lambda: clock[0], lease_renewal_seconds=0.001, batch_limit=1
    )
    task = asyncio.create_task(scheduler.run_once(force=True))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if lose_ownership:
            clock[0] += timedelta(seconds=301)
            replacement = store.claim_jobs(owner="replacement", now=clock[0])[0]
            assert await asyncio.wait_for(task, 5) == 0
            assert scheduler.last_error == "JobLeaseLostError"
            assert store.get_job(replacement.id) == replacement
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            clock[0] += timedelta(seconds=301)
            assert store.claim_jobs(owner="replacement", now=clock[0])[0].attempt_count == 2
        assert cancelled.is_set()
        assert not scheduler.health()["running"]
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        store.close()


@pytest.mark.asyncio
async def test_failure_after_renewal_stops_heartbeat_and_retries_once(tmp_path, monkeypatch):
    store = SQLiteStore(str(tmp_path / "failure.sqlite"))
    start = datetime.now(timezone.utc)
    clock = [start]
    job = DurableJob(kind="test", available_at=start)
    store.enqueue_job(job)
    renewed = asyncio.Event()
    original = store.renew_job_lease
    calls = []

    def renew(*args, **kwargs):
        calls.append(kwargs["now"])
        result = original(*args, **kwargs)
        renewed.set()
        return result

    monkeypatch.setattr(store, "renew_job_lease", renew)

    async def process(claimed):
        clock[0] += timedelta(seconds=200)
        await renewed.wait()
        raise ConnectionError("provider unavailable")

    scheduler = MaintenanceScheduler(
        store, process, clock=lambda: clock[0], lease_renewal_seconds=0.001, batch_limit=1
    )
    try:
        assert await asyncio.wait_for(scheduler.run_once(force=True), 5) == 0
        saved = store.get_job(job.id)
        assert saved.status == "pending"
        assert saved.attempt_count == 1
        assert saved.last_error == "ConnectionError"
        assert len(calls) == 1
        assert not original(
            job.id, owner=scheduler.owner, lease_token=saved.lease_token, now=clock[0]
        )
        assert store.get_job(job.id) == saved
    finally:
        store.close()


@pytest.mark.asyncio
async def test_noncooperative_processor_is_fenced_after_cancellation_grace(tmp_path):
    store = SQLiteStore(str(tmp_path / "stubborn.sqlite"))
    start = datetime.now(timezone.utc)
    store.enqueue_job(DurableJob(kind="test", available_at=start))
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    commits = []

    async def process(job):
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass
        commits.append(
            store.complete_job(
                job.id, owner=job.lease_owner, lease_token=job.lease_token, now=start
            )
        )
        finished.set()

    scheduler = MaintenanceScheduler(
        store, process, clock=lambda: start, processor_cancel_seconds=0.001, batch_limit=1
    )
    task = asyncio.create_task(scheduler.run_once(force=True))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        saved = store.list_jobs()[0]
        assert saved.status == "pending"
        assert saved.last_error == "processor_cancellation_timeout"
        assert saved.lease_token is None
        release.set()
        await asyncio.wait_for(finished.wait(), 5)
        assert commits == [False]
        assert store.get_job(saved.id) == saved
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        store.close()
