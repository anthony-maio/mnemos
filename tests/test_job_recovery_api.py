from datetime import datetime, timedelta, timezone

import pytest

from mnemos.jobs import DurableJob
from tests.test_memory_service import service


@pytest.mark.asyncio
async def test_scoped_retry_recovers_exhausted_write_without_duplicate_fact(tmp_path):
    memory, context = service(tmp_path)
    data = {"context": context.model_dump()}
    try:
        accepted = await memory.dispatch(
            "remember", {**data, "content": "Use SQLite for local memory"}
        )
        job_id = accepted["job_id"]
        for attempt in range(5):
            now = datetime.now(timezone.utc) + timedelta(minutes=attempt * 2)
            claimed = memory.engine.store.claim_jobs(owner="outage", now=now)[0]
            assert memory.engine.store.fail_job(
                job_id,
                owner="outage",
                lease_token=claimed.lease_token,
                error="ConnectionError",
                now=now,
            )
        exhausted = await memory.dispatch("jobs", {**data, "status": "exhausted", "limit": 10})
        assert [item["id"] for item in exhausted["jobs"]] == [job_id]
        assert "payload" not in exhausted["jobs"][0]
        assert "lease_token" not in exhausted["jobs"][0]
        retried = await memory.dispatch("retry_job", {**data, "job_id": job_id})
        assert retried["retried"] is True
        assert retried["job"]["status"] == "pending"
        assert retried["job"]["attempt_count"] == 0
        with pytest.raises(ValueError, match="exhausted"):
            await memory.dispatch("retry_job", {**data, "job_id": job_id})
        await memory.maintenance.run_once(force=True)
        assert len(memory.engine.store.get_all()) == 1
        assert memory.engine.store.get_job(job_id).status == "completed"
        with pytest.raises(ValueError, match="exhausted"):
            await memory.dispatch("retry_job", {**data, "job_id": job_id})
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("partition", ["user_profile_id", "project_id", "workspace_id"])
async def test_job_list_filters_scope_before_limit_and_retry_rejects_foreign_job(
    tmp_path, partition
):
    memory, context = service(tmp_path)
    data = {"context": context.model_dump()}
    try:
        foreign_context = {**context.model_dump(), partition: "other"}
        foreign = memory.engine.store.enqueue_job(
            DurableJob(
                kind="extract",
                status="exhausted",
                payload={"context": foreign_context},
            )
        )
        own = memory.engine.store.enqueue_job(
            DurableJob(
                kind="extract",
                status="exhausted",
                payload={"context": context.model_dump()},
            )
        )
        response = await memory.dispatch("jobs", {**data, "status": "exhausted", "limit": 1})
        assert [item["id"] for item in response["jobs"]] == [own.id]
        snapshot = await memory.dispatch("list", data)
        assert snapshot["jobs"]["exhausted"] == 1
        with pytest.raises(KeyError):
            await memory.dispatch("retry_job", {**data, "job_id": foreign.id})
        assert memory.engine.store.get_job(foreign.id).status == "exhausted"
        with pytest.raises(ValueError, match="status"):
            await memory.dispatch("jobs", {**data, "status": "invented"})
    finally:
        memory.engine.store.close()


def test_cli_and_ui_expose_scoped_job_recovery(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    from mnemos.client import MnemosClient
    from mnemos.control_plane import ControlPlaneService
    from mnemos.service import MnemosService
    from mnemos.ui_server import MnemosUiRouter

    memory, context = service(tmp_path)
    registry_path = tmp_path / "scopes.json"
    memory.registry.save(registry_path)
    job = memory.engine.store.enqueue_job(
        DurableJob(
            kind="extract",
            status="exhausted",
            payload={"context": context.model_dump()},
        )
    )
    runtime = MnemosService(dispatcher=memory, descriptor_path=tmp_path / "runtime.json")
    runtime.start()
    env = {
        **os.environ,
        "MNEMOS_RUNTIME_DESCRIPTOR": str(runtime.descriptor_path),
        "MNEMOS_SCOPE_REGISTRY": str(registry_path),
        "MNEMOS_USER_PROFILE_ID": context.user_profile_id,
        "MNEMOS_PROJECT_ROOT": str(tmp_path),
    }

    def cli(*args):
        result = subprocess.run(
            [sys.executable, "-m", "mnemos.cli", *args],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
        return json.loads(result.stdout)

    try:
        assert cli("jobs", "--status", "exhausted")["jobs"][0]["id"] == job.id
        assert cli("job-status", job.id)["job"]["status"] == "exhausted"
        control = ControlPlaneService(
            cwd=tmp_path,
            env=env,
            client=MnemosClient(descriptor_path=runtime.descriptor_path, auto_start=False),
        )
        router = MnemosUiRouter(control)
        response = router.handle("GET", "/api/jobs/exhausted", None)
        assert json.loads(response.body)["jobs"][0]["id"] == job.id
        legacy_response = router.handle("GET", "/api/legacy-review?limit=10", None)
        assert legacy_response.status == 200
        legacy_payload = json.loads(legacy_response.body)
        assert legacy_payload["read_only"] is True
        assert legacy_payload["facts"] == []
        headers = {"Host": "127.0.0.1", "X-Mnemos-CSRF": router.csrf_token}
        body = json.dumps({"job_id": job.id}).encode()
        assert router.handle("POST", "/api/jobs/retry", body, {"Host": "127.0.0.1"}).status == 403
        response = router.handle("POST", "/api/jobs/retry", body, headers)
        assert response.status == 200
        assert json.loads(response.body)["retried"] is True
        assert cli("job-status", job.id)["job"]["status"] == "pending"
        assert cli("legacy-review")["facts"] == []
        another = memory.engine.store.enqueue_job(
            DurableJob(
                kind="extract",
                status="exhausted",
                payload={"context": context.model_dump()},
            )
        )
        assert cli("retry-job", another.id)["retried"] is True
    finally:
        runtime.stop()
