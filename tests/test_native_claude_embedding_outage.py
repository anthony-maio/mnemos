"""Deterministic gates for the native Claude embedding-outage protocol."""

from __future__ import annotations

import socket
import json
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.native_hosts.claude_embedding_outage_v1 import (
    TARGET_VECTOR_DIMENSION,
    CLAUDE_PREFIX,
    owned_non_listening_socket,
    require_retrieval_response,
    require_native_retrieval,
    require_seed_unchanged,
    synthetic_vector,
)


def _response(*, foreign: bool = False, semantic: object = None) -> dict[str, object]:
    target = {
        "id": "target-maple-heron-719",
        "content": "The deployment marker is maple-heron-719.",
        "scope": "project",
        "scope_id": "project-1",
        "scores": {
            "semantic": semantic,
            "state_status": "not_measured",
            "graph_status": "not_used",
            "total": None,
            "retrieval_mode": "lexical_only",
        },
    }
    facts = [target]
    if foreign:
        facts.append(
            {
                "id": "foreign-canary-849",
                "content": "Foreign deployment marker is not project memory.",
                "scope": "project",
                "scope_id": "foreign-project",
                "scores": {
                    "semantic": None,
                    "state_status": "not_measured",
                    "graph_status": "not_used",
                    "total": None,
                    "retrieval_mode": "lexical_only",
                },
            }
        )
    return {
        "status": "degraded",
        "degraded": True,
        "retrieval": {"mode": "lexical_only", "reason": "embedding_unavailable"},
        "facts": facts,
        "context": "[target-maple-heron-719] The deployment marker is maple-heron-719.\n",
    }


def test_synthetic_vector_is_explicit_nonzero_schema_fixture() -> None:
    vector = synthetic_vector()
    assert len(vector) == TARGET_VECTOR_DIMENSION
    assert vector[0] == 1.0
    assert sum(abs(value) for value in vector) == 1.0


def test_owned_socket_is_bound_but_not_listening() -> None:
    sock = owned_non_listening_socket()
    try:
        host, port = sock.getsockname()
        assert host == "127.0.0.1"
        with socket.socket() as probe:
            probe.settimeout(0.2)
            assert probe.connect_ex((host, port)) != 0
    finally:
        sock.close()


def test_retrieval_gate_accepts_degraded_lexical_target() -> None:
    require_retrieval_response(
        _response(),
        target_id="target-maple-heron-719",
        target_content="The deployment marker is maple-heron-719.",
        project_id="project-1",
        foreign_id="foreign-canary-849",
        foreign_content="Foreign deployment marker is not project memory.",
    )


def native_records():
    prompt = "What is the deployment marker for this project?"
    records = [
        {
            "type": "user",
            "sessionId": "session-1",
            "message": {"content": prompt},
        },
        {
            "type": "assistant",
            "sessionId": "session-1",
            "message": {
                "model": "claude-sonnet-4-6",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": CLAUDE_PREFIX + "mnemos_retrieve",
                        "input": {"query": "deployment marker"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "sessionId": "session-1",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": json.dumps({"result": json.dumps(_response())}),
                    }
                ]
            },
        },
        {
            "type": "assistant",
            "sessionId": "session-1",
            "message": {
                "model": "claude-sonnet-4-6",
                "content": "The marker is maple-heron-719.",
            },
        },
    ]
    return records


def test_native_gate_requires_correlated_retrieval_and_canary_answer() -> None:
    records = native_records()
    prompt = records[0]["message"]["content"]
    assert (
        require_native_retrieval(
            records,
            submitted_prompt=prompt,
            target_content="The deployment marker is maple-heron-719.",
            expected_model="claude-sonnet-4-6",
            expected_query="deployment marker",
        )["tool_use_id"]
        == "tool-1"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"status": "observed"},
        {"retrieval": {"retrieval_mode": "semantic_and_lexical"}},
        {"facts": []},
        {"foreign": True},
        {"semantic": 0.5},
    ],
)
def test_retrieval_gate_rejects_unproven_or_cross_scope_response(
    change: dict[str, object],
) -> None:
    response = _response(
        foreign=bool(change.get("foreign")),
        semantic=change.get("semantic"),
    )
    for key, value in change.items():
        if key == "foreign" or key == "semantic":
            continue
        response[key] = value  # type: ignore[literal-required]
    with pytest.raises(ValueError):
        require_retrieval_response(
            response,
            target_id="target-maple-heron-719",
            target_content="The deployment marker is maple-heron-719.",
            project_id="project-1",
            foreign_id="foreign-canary-849",
            foreign_content="Foreign deployment marker is not project memory.",
        )


def test_seed_identity_gate_requires_content_version_and_scope() -> None:
    class Store:
        def __init__(self, value: object) -> None:
            self.value = value

        def get(self, _fact_id: str) -> object:
            return self.value

    class Fact:
        content = "The deployment marker is maple-heron-719."
        version = 1
        metadata = {
            "scope": "project",
            "scope_id": "project-1",
            "user_profile_id": "profile-1",
            "source_kind": "synthetic_fixture",
            "assertion_status": "asserted",
        }

    expected = {
        "id": {
            "content": Fact.content,
            "version": 1,
            "metadata": dict(Fact.metadata),
        }
    }
    require_seed_unchanged(Store(Fact()), expected)
    Fact.version = 2
    with pytest.raises(ValueError):
        require_seed_unchanged(Store(Fact()), expected)


@pytest.mark.asyncio
async def test_gate_accepts_actual_local_service_lexical_fallback(tmp_path: Path) -> None:
    from tests.test_memory_service import service
    from tests.test_retrieval_provider_outage import OfflineEmbeddings, seed

    memory, context = service(tmp_path)
    memory.engine._embedder = OfflineEmbeddings()
    try:
        target = seed(memory, context, "The deployment marker is maple-heron-719.")
        response = await memory.dispatch(
            "retrieve",
            {"context": context.model_dump(), "query": "deployment marker", "top_k": 1},
        )
        require_retrieval_response(
            response,
            target_id=target.id,
            target_content=target.content,
            project_id=context.project_id or "",
            foreign_id="foreign-canary-849",
            foreign_content="Foreign deployment marker is not project memory.",
        )
    finally:
        memory.engine.store.close()


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_tool",
        "other_tool",
        "missing_model",
        "foreign_session",
        "tool_error",
        "early_answer_only",
        "extra_prompt",
        "duplicate_result",
    ],
)
def test_native_gate_rejects_unproven_tool_conversation(defect):
    records = native_records()
    if defect == "wrong_tool":
        records[1]["message"]["content"][0]["name"] += "_fake"
    elif defect == "other_tool":
        records[1]["message"]["content"].append(
            {"type": "tool_use", "name": "Bash", "id": "extra", "input": {}}
        )
    elif defect == "missing_model":
        records[1]["message"].pop("model")
        records[3]["message"].pop("model")
    elif defect == "foreign_session":
        records[2]["sessionId"] = "another-session"
    elif defect == "tool_error":
        records[2]["message"]["content"][0]["is_error"] = True
    elif defect == "early_answer_only":
        records[1]["message"]["content"].insert(0, {"type": "text", "text": "maple-heron-719"})
        records[3]["message"]["content"] = "I could not retrieve a fact."
    elif defect == "extra_prompt":
        extra = deepcopy(records[0])
        extra["message"]["content"] = "Try again."
        records.insert(1, extra)
    elif defect == "duplicate_result":
        records.insert(3, deepcopy(records[2]))
    with pytest.raises(ValueError):
        require_native_retrieval(
            records,
            submitted_prompt=records[0]["message"]["content"],
            target_content="The deployment marker is maple-heron-719.",
            expected_model="claude-sonnet-4-6",
        )


def test_read_only_seed_projection_matches_actual_sqlite(tmp_path):
    from mnemos.utils.storage import SQLiteStore
    from mnemos.types import MemoryChunk
    from benchmarks.native_hosts.claude_embedding_outage_v1 import read_only_seed_summary

    database = (tmp_path / "memory.sqlite").resolve()
    chunk = MemoryChunk(
        content="deployment marker fixture",
        embedding=synthetic_vector(),
        metadata={"scope": "project", "scope_id": "project-1", "assertion_status": "asserted"},
    )
    store = SQLiteStore(str(database))
    try:
        store.store(chunk)
    finally:
        store.close()
    observed = read_only_seed_summary(database, [chunk.id])
    assert observed == {
        chunk.id: {"content": chunk.content, "version": chunk.version, "metadata": chunk.metadata}
    }


def test_native_result_decodes_to_recorded_service_payload():
    rows = native_records()
    result = require_native_retrieval(
        rows,
        submitted_prompt=rows[0]["message"]["content"],
        target_content="The deployment marker is maple-heron-719.",
        expected_model="claude-sonnet-4-6",
        expected_session_id="session-1",
    )
    assert result["response"] == _response()
    with pytest.raises(ValueError, match="launched"):
        require_native_retrieval(
            rows,
            submitted_prompt=rows[0]["message"]["content"],
            target_content="The deployment marker is maple-heron-719.",
            expected_model="claude-sonnet-4-6",
            expected_session_id="wrong-session",
        )
