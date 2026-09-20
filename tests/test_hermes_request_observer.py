from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterator

import httpx
import pytest

from benchmarks.native_hosts.hermes_request_observer import observe_requests

ENDPOINT = "http://127.0.0.1:11434/v1/chat/completions"


def test_observer_captures_exact_unicode_body_and_safe_metadata(tmp_path: Path) -> None:
    payload = {
        "model": "fixture",
        "messages": [{"role": "user", "content": "caf\u00e9 \N{SNOWMAN}"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "mnemos_forget_related",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selected_fact_ids": {"type": "array", "items": {"type": "string"}},
                            "selected_source_event_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["selected_fact_ids", "selected_source_event_ids"],
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "mnemos_forget_related"}},
        "tool_arguments_example": {
            "selected_fact_ids": [],
            "selected_source_event_ids": [],
        },
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    expected_response = httpx.Response(200)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.content == body
        expected_response.request = request
        return expected_response

    with observe_requests(ENDPOINT, tmp_path):
        response = httpx.Client(
            transport=httpx.MockTransport(handler),
            headers={"Authorization": "Bearer secret", "X-Secret": "private"},
        ).post(ENDPOINT, content=body)
    assert response is expected_response
    assert response.status_code == 200
    files = sorted(tmp_path.iterdir())
    assert [path.suffix for path in files] == [".bin", ".json"]
    metadata = json.loads(next(path for path in files if path.suffix == ".json").read_text())
    assert metadata["method"] == "POST"
    assert metadata["url"] == ENDPOINT
    assert metadata["byte_count"] == len(body)
    assert (tmp_path / metadata["body_file"]).read_bytes() == body
    assert metadata["sha256"] == hashlib.sha256(body).hexdigest()
    assert "Authorization" not in json.dumps(metadata)
    assert "secret" not in json.dumps(metadata)


def test_nonmatching_request_is_untouched_and_not_recorded(tmp_path: Path) -> None:
    other = "http://127.0.0.1:11434/api/generate"

    with observe_requests(ENDPOINT, tmp_path):
        with httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(201, request=request))
        ) as client:
            response = client.post(other, json={"prompt": "hello"})
    assert response.status_code == 201
    assert list(tmp_path.iterdir()) == []


def test_artifact_created_after_factory_is_preserved_at_entry(tmp_path: Path) -> None:
    original = httpx.Client.send
    observer = observe_requests(ENDPOINT, tmp_path)
    artifact = tmp_path / "request-0002.json"
    artifact.write_bytes(b"prior metadata")
    with pytest.raises(ValueError, match="already contains"):
        with observer:
            pytest.fail("must not enter with existing evidence")
    assert artifact.read_bytes() == b"prior metadata"
    assert httpx.Client.send is original


def test_escaping_exception_restores_patch(tmp_path: Path) -> None:
    original = httpx.Client.send
    error = RuntimeError("abort observation")
    with pytest.raises(RuntimeError) as raised:
        with observe_requests(ENDPOINT, tmp_path):
            assert httpx.Client.send is not original
            raise error
    assert raised.value is error
    assert httpx.Client.send is original


def test_exception_passes_through_and_patch_is_restored(tmp_path: Path) -> None:
    original_send = httpx.Client.send
    expected = RuntimeError("fixture transport failure")

    def handler(request: httpx.Request) -> httpx.Response:
        raise expected

    with observe_requests(ENDPOINT, tmp_path):
        assert httpx.Client.send is not original_send
        with pytest.raises(RuntimeError) as raised:
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                client.post(ENDPOINT, json={"hello": "world"})
        assert raised.value is expected
    assert httpx.Client.send is original_send
    assert len(list(tmp_path.iterdir())) == 2


def test_matching_streamed_request_fails_without_consuming_stream(tmp_path: Path) -> None:
    class Stream(httpx.SyncByteStream):
        consumed = False

        def __iter__(self) -> Iterator[bytes]:
            self.consumed = True
            yield b"streamed"

    stream = Stream()
    with observe_requests(ENDPOINT, tmp_path):
        with httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200))
        ) as client:
            with pytest.raises(ValueError, match="streamed"):
                client.post(ENDPOINT, content=stream)
    assert stream.consumed is False
    assert list(tmp_path.iterdir()) == []


def test_existing_request_artifact_is_rejected_without_patching(tmp_path: Path) -> None:
    artifact = tmp_path / "request-0001.bin"
    artifact.write_bytes(b"keep")
    original_send = httpx.Client.send
    with pytest.raises(ValueError, match="request artifacts"):
        observe_requests(ENDPOINT, tmp_path)
    assert httpx.Client.send is original_send
    assert artifact.read_bytes() == b"keep"


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://user:password@127.0.0.1:11434/v1/chat/completions",
        "http://127.0.0.1:11434/v1/chat/completions?token=secret",
        "http://127.0.0.1:11434/v1/chat/completions#fragment",
    ],
)
def test_endpoint_rejects_credentials_query_and_fragment(endpoint: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        observe_requests(endpoint, tmp_path)
