"""Provider outages wait without spending attempts; deterministic rejections stop at once."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from mnemos.jobs import DurableJob
from mnemos.utils.storage import SQLiteStore
from tests.test_suppression_ordering import _remember, _service, _settle

T0 = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)


def _store_with_job(tmp_path: Path, *, max_attempts: int = 5) -> SQLiteStore:
    store = SQLiteStore(str(tmp_path / "jobs.sqlite"))
    store.enqueue_job(
        DurableJob(id="job", kind="extract", payload={}, max_attempts=max_attempts, available_at=T0)
    )
    return store


def test_provider_failures_back_off_without_spending_attempts(tmp_path: Path) -> None:
    store = _store_with_job(tmp_path)
    try:
        now = T0
        delays = []
        for _ in range(9):
            job = store.claim_jobs(owner="worker", now=now)[0]
            assert store.fail_job(
                job.id,
                owner="worker",
                lease_token=job.lease_token,
                error="ConnectionError",
                failure_class="provider",
                now=now,
            )
            current = store.get_job(job.id)
            assert current is not None
            assert (current.status, current.attempt_count) == ("pending", 0)
            delays.append((current.available_at - now).total_seconds())
            now = current.available_at
        assert delays == [15, 30, 60, 120, 240, 480, 900, 900, 900]
    finally:
        store.close()


def test_terminal_failure_exhausts_immediately_with_its_reason(tmp_path: Path) -> None:
    store = _store_with_job(tmp_path)
    try:
        job = store.claim_jobs(owner="worker", now=T0)[0]
        assert store.fail_job(
            job.id,
            owner="worker",
            lease_token=job.lease_token,
            error="version_conflict",
            failure_class="terminal",
            now=T0,
        )
        current = store.get_job(job.id)
        assert current is not None
        assert (current.status, current.attempt_count, current.last_error) == (
            "exhausted",
            1,
            "version_conflict",
        )
    finally:
        store.close()


def test_expired_lease_records_why_the_job_stopped(tmp_path: Path) -> None:
    store = _store_with_job(tmp_path, max_attempts=1)
    try:
        assert store.claim_jobs(owner="crashed-worker", lease_seconds=5, now=T0)
        assert store.claim_jobs(owner="next-worker", now=T0 + timedelta(seconds=10)) == []
        current = store.get_job("job")
        assert current is not None
        assert (current.status, current.last_error) == ("exhausted", "lease_expired")
    finally:
        store.close()


async def test_missing_model_leaves_capture_pending_instead_of_exhausting(
    tmp_path: Path,
) -> None:
    memory, context = _service(tmp_path)
    store = memory.engine.store
    memory.engine.config.modules.sleep = True
    now = [datetime.now(timezone.utc) + timedelta(seconds=1)]
    memory.maintenance.clock = lambda: now[0]
    try:
        receipt = await memory.dispatch(
            "capture",
            {
                "context": context,
                "event_id": "waits-for-a-model",
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "role": "user",
                "source_kind": "codex_user_prompt",
                "content": "Deploys go out on Thursdays after the release review.",
            },
        )
        for _ in range(12):
            await memory.maintenance.run_once(force=True)
            now[0] += timedelta(minutes=16)
        job = store.get_job(receipt["job_id"])
        assert job is not None
        assert (job.status, job.attempt_count) == ("pending", 0)
        assert job.last_error == "MnemosConfigurationError"
    finally:
        store.close()


async def test_stale_explicit_correction_stops_after_one_attempt(tmp_path: Path) -> None:
    memory, context = _service(tmp_path)
    store = memory.engine.store
    try:
        await _remember(memory, context, "npm", "Use npm")
        await _settle(memory)
        fact = store.get_all()[0]
        corrections = [
            await memory.dispatch(
                "correct",
                {
                    "context": context,
                    "event_id": event_id,
                    "fact_id": fact.id,
                    "expected_version": fact.version,
                    "content": content,
                },
            )
            for event_id, content in (("first", "Use pnpm"), ("second", "Use yarn"))
        ]
        await _settle(memory)
        current = store.get(fact.id)
        assert current is not None and current.content == "Use pnpm"
        stale = store.get_job(corrections[1]["job_id"])
        assert stale is not None
        assert (stale.status, stale.attempt_count, stale.last_error) == (
            "exhausted",
            1,
            "version_conflict",
        )
    finally:
        store.close()
