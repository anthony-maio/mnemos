"""The prefix probe retains wire evidence and refuses unverified embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.local_memory.embedding_prefix_live_v1 import RecordedAPI, RecordedProvider
from benchmarks.local_memory.prepare_source_association_vectors import DIGEST, MODEL


class Transport:
    def __init__(self, gpu: int = 1024, status: int = 200) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.gpu = gpu
        self.status = status

    def __call__(
        self, method: str, url: str, body: bytes | None, timeout: int
    ) -> tuple[int, dict[str, str], bytes]:
        assert timeout == 60
        result: dict[str, Any]
        if url.endswith("/embed"):
            payload = json.loads(body or b"null")
            self.payloads.append(payload)
            result = {"embeddings": [[1.0] + [0.0] * 767 for _ in payload["input"]]}
        else:
            assert method == "GET" and url.endswith("/ps")
            result = {
                "models": [
                    {"name": MODEL, "digest": DIGEST, "context_length": 2048, "size_vram": self.gpu}
                ]
            }
        return self.status, {"Content-Type": "application/json"}, json.dumps(result).encode()


def test_records_exact_payloads_and_batches_without_retries(tmp_path: Path) -> None:
    transport = Transport()
    api = RecordedAPI(tmp_path, transport=transport)
    provider = RecordedProvider(api, "unprefixed")
    texts = [f"document-{i}" for i in range(17)]
    assert len(provider.embed_batch(texts)) == 17
    assert [p["input"] for p in transport.payloads] == [texts[:16], texts[16:]]
    assert all(
        p == {"model": MODEL, "input": p["input"], "truncate": False} for p in transport.payloads
    )
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert all(r["arm"] == "unprefixed" and r["status"] == "complete" for r in rows)
    assert json.loads((tmp_path / rows[0]["request_file"]).read_bytes()) == transport.payloads[0]
    assert len(provider.loaded_profiles) == 2


def test_cpu_only_response_fails_and_remains_recorded(tmp_path: Path) -> None:
    provider = RecordedProvider(RecordedAPI(tmp_path, transport=Transport(gpu=0)), "nomic_task")
    with pytest.raises(ValueError, match="GPU"):
        provider.embed("search_query: example")
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert (tmp_path / rows[-1]["response_file"]).is_file()


def test_http_failure_has_raw_evidence_and_no_retry(tmp_path: Path) -> None:
    transport = Transport(status=500)
    provider = RecordedProvider(RecordedAPI(tmp_path, transport=transport), "unprefixed")
    with pytest.raises(RuntimeError, match="HTTP 500"):
        provider.embed("example")
    assert len(transport.payloads) == 1
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["status"] == "failed"
    assert rows[0]["response_status"] == 500
    assert (tmp_path / rows[0]["response_file"]).is_file()
