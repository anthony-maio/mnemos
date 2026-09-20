"""Missing and inaccessible jobs share a non-disclosing HTTP error."""

import httpx
import pytest

from mnemos.config import MnemosConfig, ModuleConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.service import MnemosService
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore


@pytest.fixture
def job_service(tmp_path):
    registry = ScopeRegistry()
    contexts = []
    for name, profile in (("owner", "local"), ("foreign", "local"), ("other", "other")):
        project = tmp_path / name
        project.mkdir()
        registry.register_project(project, user_profile_id=profile, enabled=True)
        contexts.append(
            registry.resolve(
                project, user_profile_id=profile, session_id="test", host="codex"
            ).model_dump(mode="json")
        )
    store = SQLiteStore(str(tmp_path / "memory.sqlite"))
    engine = MnemosEngine(
        config=MnemosConfig(
            modules=ModuleConfig(**{name: False for name in ModuleConfig.model_fields})
        ),
        store=store,
        embedder=SimpleEmbeddingProvider(),
        llm=UnavailableLLMProvider(),
    )
    service = MnemosService(
        dispatcher=LocalMemoryService(engine, registry),
        descriptor_path=tmp_path / "runtime.json",
    )
    service.start()
    try:
        yield service, store, contexts
    finally:
        service.stop()


@pytest.mark.parametrize("operation", ["job", "retry_job"])
def test_missing_and_cross_scope_jobs_return_same_404(job_service, operation):
    service, store, contexts = job_service
    accepted = service.submit(
        "remember", {"context": contexts[0], "content": "The release label is cedar."}
    )
    job_id = accepted["job_id"]
    before = store.get_job(job_id).model_dump(mode="json")
    descriptor = service.descriptor
    with httpx.Client(trust_env=False, timeout=5) as client:
        url = f"http://127.0.0.1:{descriptor['port']}/v1/dispatch"
        headers = {"Authorization": f"Bearer {descriptor['token']}"}
        valid = client.post(
            url,
            headers=headers,
            json={"operation": "job", "payload": {"context": contexts[0], "job_id": job_id}},
        )
        assert valid.status_code == 200
        assert valid.json()["job"]["id"] == job_id
        assert "payload" not in valid.json()["job"]
        wrong_id = job_id[:-1] + ("0" if job_id[-1] != "0" else "1")
        for context, requested_id in (
            (contexts[0], wrong_id),
            (contexts[1], job_id),
            (contexts[2], job_id),
        ):
            response = client.post(
                url,
                headers=headers,
                json={
                    "operation": operation,
                    "payload": {"context": context, "job_id": requested_id},
                },
            )
            assert response.status_code == 404
            assert response.json() == {"schema_version": 1, "error": "job_not_found"}
            with pytest.raises(KeyError):
                service.submit(operation, {"context": context, "job_id": requested_id})
    assert store.get_job(job_id).model_dump(mode="json") == before


def test_unexpected_job_lookup_key_error_remains_a_server_failure(job_service, monkeypatch):
    service, store, contexts = job_service

    def broken_lookup(job_id):
        raise KeyError("private internal state")

    monkeypatch.setattr(store, "get_job", broken_lookup)
    descriptor = service.descriptor
    with httpx.Client(trust_env=False, timeout=5) as client:
        response = client.post(
            f"http://127.0.0.1:{descriptor['port']}/v1/dispatch",
            headers={"Authorization": f"Bearer {descriptor['token']}"},
            json={"operation": "job", "payload": {"context": contexts[0], "job_id": "x"}},
        )
    assert response.status_code == 500
    assert response.json() == {"schema_version": 1, "error": "dispatch_failed"}
