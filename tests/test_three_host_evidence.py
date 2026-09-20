from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.native_hosts.three_host_evidence import (
    hermes_evidence,
    require_scope_exclusion,
)

MODEL = "test-model"
DIGEST = "d" * 64


def _trace(*, repeated: bool = False, foreign: bool = False) -> list[dict]:
    args = {"query": "release"}
    first = {
        "operation": "model_response",
        "tool_calls": [
            {
                "id": "response-1:p1",
                "provider_call_id": "p1",
                "name": "mnemos_retrieve",
                "arguments": json.dumps(args),
            }
        ],
        "content": None,
    }
    result = {"schema_version": 1, "facts": []}
    if foreign:
        result["facts"] = [
            {"id": "bad-fact", "scope_id": "bad-scope", "content": "x", "version": 1}
        ]
    rows = [
        {
            "operation": "model_request",
            "model": MODEL,
            "reasoning_effort": "none",
            "options": {"num_ctx": 65536},
            "contains_foreign_label": False,
        },
        first,
        {"operation": "tool_result", "name": "mnemos_retrieve", "args": args, "result": result},
    ]
    if repeated:
        rows.extend(
            [
                {
                    "operation": "model_response",
                    "tool_calls": [
                        {
                            "id": "response-2:p2",
                            "provider_call_id": "p2",
                            "name": "mnemos_retrieve",
                            "arguments": json.dumps(args),
                        }
                    ],
                    "content": None,
                },
                {
                    "operation": "tool_result",
                    "name": "mnemos_retrieve",
                    "args": args,
                    "result": result,
                },
            ]
        )
    rows.append({"operation": "model_response", "tool_calls": [], "content": "done"})
    count = 3 if repeated else 2
    for n in range(1, count + 1):
        rows.extend(
            [
                {
                    "operation": "loaded_models_after_response",
                    "response_number": n,
                    "result": {
                        "models": [
                            {
                                "name": MODEL,
                                "digest": DIGEST,
                                "context_length": 65536,
                                "size_vram": 1,
                            }
                        ]
                    },
                },
                {"operation": "loaded_profile_verified", "response_number": n},
            ]
        )
    return rows


def _write_wire(root: Path, trace: list[dict], *, altered: bool = False) -> None:
    messages = []
    for index, response in enumerate(
        row for row in trace if row.get("operation") == "model_response"
    ):
        if not response.get("tool_calls"):
            continue
        call = response["tool_calls"][0]
        result = next(row for row in trace[index + 1 :] if row.get("operation") == "tool_result")
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call["provider_call_id"],
                            "function": {"name": call["name"], "arguments": call["arguments"]},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call["provider_call_id"],
                    "content": json.dumps(result["result"]),
                },
            ]
        )
    if altered:
        messages[0]["tool_calls"][0]["function"]["arguments"] = '{"query":"changed"}'
    root = root / "provider-requests"
    root.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"messages": messages}).encode()
    (root / "request-0001.bin").write_bytes(body)
    (root / "request-0001.json").write_text(
        json.dumps(
            {
                "body_file": "request-0001.bin",
                "byte_count": len(body),
                "method": "POST",
                "url": "http://127.0.0.1:11434/v1/chat/completions",
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def test_valid_and_sequential_repeated_calls(tmp_path: Path) -> None:
    trace = _trace(repeated=True)
    _write_wire(tmp_path, trace)
    first_body = (tmp_path / "provider-requests/request-0001.bin").read_bytes()
    (tmp_path / "provider-requests/request-0002.bin").write_bytes(first_body)
    metadata = json.loads(
        (tmp_path / "provider-requests/request-0001.json").read_text(encoding="utf-8")
    )
    metadata["body_file"] = "request-0002.bin"
    (tmp_path / "provider-requests/request-0002.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    result = hermes_evidence(trace, tmp_path, model=MODEL, digest=DIGEST)
    assert (
        result["calls"][0]["correlation_level"]
        == "unambiguous_native_arguments_with_wire_call_id_and_result"
    )


@pytest.mark.parametrize("fault", ["altered-result", "truncated-result", "altered-later-history"])
def test_wire_result_must_remain_exact_even_with_a_valid_earlier_copy(tmp_path, fault):
    trace = _trace()
    _write_wire(tmp_path, trace)
    root = tmp_path / "provider-requests"
    metadata = json.loads((root / "request-0001.json").read_text())
    body = json.loads((root / "request-0001.bin").read_bytes())
    body["messages"][1]["content"] = (
        '{"schema_version":1'
        if fault == "truncated-result"
        else json.dumps({"schema_version": 1, "facts": [{"id": "invented"}]})
    )
    ordinal = "0002" if fault == "altered-later-history" else "0001"
    raw = json.dumps(body).encode()
    metadata.update(
        body_file=f"request-{ordinal}.bin",
        byte_count=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    (root / metadata["body_file"]).write_bytes(raw)
    (root / f"request-{ordinal}.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        hermes_evidence(trace, tmp_path, model=MODEL, digest=DIGEST)


@pytest.mark.parametrize("fault", ["missing-foreign-check", "call-bearing-final", "empty-final"])
def test_final_response_and_input_exclusion_are_required(tmp_path, fault):
    trace = _trace()
    _write_wire(tmp_path, trace)
    if fault == "missing-foreign-check":
        next(row for row in trace if row["operation"] == "model_request").pop(
            "contains_foreign_label"
        )
    else:
        final = [row for row in trace if row["operation"] == "model_response"][-1]
        if fault == "empty-final":
            final["content"] = ""
        else:
            final["tool_calls"] = [
                {
                    "id": "response-2:pending",
                    "provider_call_id": "pending",
                    "name": "mnemos_retrieve",
                    "arguments": '{"query":"pending"}',
                }
            ]
    with pytest.raises(ValueError):
        hermes_evidence(trace, tmp_path, model=MODEL, digest=DIGEST)


def test_rejects_ambiguous_pending_calls(tmp_path: Path) -> None:
    trace = _trace()
    trace[1]["tool_calls"].append(
        {
            "id": "response-1:p2",
            "provider_call_id": "p2",
            "name": "mnemos_retrieve",
            "arguments": json.dumps({"query": "release"}),
        }
    )
    with pytest.raises(ValueError, match="ambiguous"):
        hermes_evidence(trace, tmp_path, model=MODEL, digest=DIGEST)


@pytest.mark.parametrize("altered", [True])
def test_rejects_altered_wire_body(tmp_path: Path, altered: bool) -> None:
    trace = _trace()
    _write_wire(tmp_path, trace, altered=altered)
    with pytest.raises(ValueError, match="wire"):
        hermes_evidence(trace, tmp_path, model=MODEL, digest=DIGEST)

    _write_wire(tmp_path, trace)
    body = json.loads((tmp_path / "provider-requests/request-0001.bin").read_text(encoding="utf-8"))
    body["messages"] = body["messages"][:1]
    raw = json.dumps(body).encode()
    (tmp_path / "provider-requests/request-0001.bin").write_bytes(raw)
    metadata = json.loads(
        (tmp_path / "provider-requests/request-0001.json").read_text(encoding="utf-8")
    )
    metadata["byte_count"] = len(raw)
    metadata["sha256"] = hashlib.sha256(raw).hexdigest()
    (tmp_path / "provider-requests/request-0001.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="wire"):
        hermes_evidence(trace, tmp_path, model=MODEL, digest=DIGEST)


def test_rejects_mismatched_wire_call_and_metadata_hash(tmp_path: Path) -> None:
    trace = _trace()
    _write_wire(tmp_path, trace)
    metadata = json.loads(
        (tmp_path / "provider-requests/request-0001.json").read_text(encoding="utf-8")
    )
    metadata["sha256"] = "0" * 64
    (tmp_path / "provider-requests/request-0001.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="bytes"):
        hermes_evidence(trace, tmp_path, model=MODEL, digest=DIGEST)

    _write_wire(tmp_path, trace)
    body = json.loads((tmp_path / "provider-requests/request-0001.bin").read_text(encoding="utf-8"))
    body["messages"][0]["tool_calls"][0]["function"]["name"] = "mnemos_inspect"
    raw = json.dumps(body).encode()
    (tmp_path / "provider-requests/request-0001.bin").write_bytes(raw)
    metadata = json.loads(
        (tmp_path / "provider-requests/request-0001.json").read_text(encoding="utf-8")
    )
    metadata["byte_count"] = len(raw)
    metadata["sha256"] = hashlib.sha256(raw).hexdigest()
    (tmp_path / "provider-requests/request-0001.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="wire"):
        hermes_evidence(trace, tmp_path, model=MODEL, digest=DIGEST)

    _write_wire(tmp_path, trace)
    body = json.loads((tmp_path / "provider-requests/request-0001.bin").read_text(encoding="utf-8"))
    body["messages"][1]["tool_call_id"] = "wrong-provider-id"
    raw = json.dumps(body).encode()
    (tmp_path / "provider-requests/request-0001.bin").write_bytes(raw)
    metadata = json.loads(
        (tmp_path / "provider-requests/request-0001.json").read_text(encoding="utf-8")
    )
    metadata["byte_count"] = len(raw)
    metadata["sha256"] = hashlib.sha256(raw).hexdigest()
    (tmp_path / "provider-requests/request-0001.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="wire"):
        hermes_evidence(trace, tmp_path, model=MODEL, digest=DIGEST)


def test_rejects_missing_loaded_check_or_wrong_digest(tmp_path: Path) -> None:
    trace = _trace()
    _write_wire(tmp_path, trace)
    trace = [row for row in trace if row.get("operation") != "loaded_profile_verified"]
    with pytest.raises(ValueError, match="loaded profile"):
        hermes_evidence(trace, tmp_path, model=MODEL, digest=DIGEST)

    trace = _trace()
    _write_wire(tmp_path, trace)
    with pytest.raises(ValueError, match="digest"):
        hermes_evidence(trace, tmp_path, model=MODEL, digest="x" * 64)


def test_scope_exclusion_checks_nested_structured_ids() -> None:
    calls = [{"result": {"facts": [{"id": "foreign-fact", "metadata": {"project_id": "foreign"}}]}}]
    with pytest.raises(ValueError, match="scope"):
        require_scope_exclusion(calls, forbidden_ids=set(), forbidden_scope_ids={"foreign"})
    with pytest.raises(ValueError, match="fact"):
        require_scope_exclusion(calls, forbidden_ids={"foreign-fact"}, forbidden_scope_ids=set())
    require_scope_exclusion(calls, forbidden_ids={"other"}, forbidden_scope_ids={"different"})
