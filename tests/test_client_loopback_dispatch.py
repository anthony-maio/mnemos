"""Dispatch accepts each advertised loopback address without changing its target."""

from pathlib import Path

import httpx
import pytest

from mnemos.client import MnemosClient, ServiceClientError


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
async def test_dispatch_preserves_loopback_target_and_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"schema_version": 1, "status": "ok"})

    client = MnemosClient(descriptor_path=tmp_path / "runtime.json", auto_start=False)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as transport:
        monkeypatch.setattr(client, "_loop_http_client", lambda: transport)
        result = await client._dispatch_http(
            "health", {}, {"host": host, "port": 43123, "token": "test-token"}
        )
    assert result == {"schema_version": 1, "status": "ok"}
    assert len(requests) == 1
    request = requests[0]
    assert request.url.host == host
    assert request.url.port == 43123
    assert request.url.path == "/v1/dispatch"
    assert request.headers["Authorization"] == "Bearer test-token"
    assert request.method == "POST"


@pytest.mark.parametrize("host", ["192.0.2.1", "example.com", "[::1]"])
async def test_dispatch_rejects_unapproved_hosts_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    client = MnemosClient(descriptor_path=tmp_path / "runtime.json", auto_start=False)

    def unexpected_transport() -> None:
        pytest.fail("rejected descriptor must not reach the transport")

    monkeypatch.setattr(client, "_loop_http_client", unexpected_transport)
    with pytest.raises(ServiceClientError, match="not loopback"):
        await client._dispatch_http(
            "health", {}, {"host": host, "port": 43123, "token": "test-token"}
        )
