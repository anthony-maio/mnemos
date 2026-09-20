"""Missing and inaccessible facts share a non-disclosing HTTP error."""

import httpx
import pytest

from tests.test_service_job_errors import job_service


def dispatch_http(service, operation, payload):
    descriptor = service.descriptor
    with httpx.Client(trust_env=False, timeout=5) as client:
        return client.post(
            f"http://127.0.0.1:{descriptor['port']}/v1/dispatch",
            headers={"Authorization": f"Bearer {descriptor['token']}"},
            json={"operation": operation, "payload": payload},
        )


@pytest.mark.parametrize("operation", ["inspect", "correct", "forget"])
def test_missing_and_cross_scope_facts_return_same_404(job_service, operation):
    service, store, contexts = job_service
    service.submit("remember", {"context": contexts[0], "content": "The release label is cedar."})
    service.submit("maintenance", {})
    target = store.get_all()[0]
    before = target.model_dump(mode="json")
    valid = dispatch_http(service, "inspect", {"context": contexts[0], "fact_id": target.id})
    assert valid.status_code == 200
    assert valid.json()["fact"]["id"] == target.id
    for context, requested_id in (
        (contexts[0], "absent-fact"),
        (contexts[1], target.id),
        (contexts[2], target.id),
    ):
        payload = {
            "context": context,
            "fact_id": requested_id,
            "expected_version": target.version,
            "content": "The release label is maple.",
        }
        response = dispatch_http(service, operation, payload)
        assert response.status_code == 404
        assert response.json() == {"schema_version": 1, "error": "fact_not_found"}
        with pytest.raises(KeyError):
            service.submit(operation, payload)
    assert store.get(target.id).model_dump(mode="json") == before
    assert len(store.get_all()) == 1


@pytest.mark.parametrize("operation", ["inspect", "correct", "forget"])
def test_erased_fact_returns_404(job_service, operation):
    service, store, contexts = job_service
    service.submit("remember", {"context": contexts[0], "content": "The release label is cedar."})
    service.submit("maintenance", {})
    target = store.get_all()[0]
    payload = {
        "context": contexts[0],
        "fact_id": target.id,
        "expected_version": target.version,
        "content": "The release label is maple.",
    }
    forgotten = dispatch_http(service, "forget", payload)
    assert forgotten.status_code == 200
    assert forgotten.json()["forgotten"] is True
    response = dispatch_http(service, operation, payload)
    assert response.status_code == 404
    assert response.json() == {"schema_version": 1, "error": "fact_not_found"}
    assert store.get_all() == []


@pytest.mark.parametrize("operation", ["inspect", "correct", "forget"])
def test_unexpected_fact_lookup_key_error_remains_server_failure(
    job_service, monkeypatch, operation
):
    service, store, contexts = job_service

    def broken_lookup(fact_id):
        raise KeyError("private internal state")

    monkeypatch.setattr(store, "get", broken_lookup)
    response = dispatch_http(
        service,
        operation,
        {
            "context": contexts[0],
            "fact_id": "absent-fact",
            "expected_version": 1,
            "content": "The release label is maple.",
        },
    )
    assert response.status_code == 500
    assert response.json() == {"schema_version": 1, "error": "dispatch_failed"}
