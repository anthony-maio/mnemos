from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.local_memory.contracts import FactRecord, SessionScenario, SourceEvent, load_dataset
from benchmarks.local_memory.harness import DEFAULT_PROFILES
from benchmarks.local_memory.ollama_adapter import OllamaEvaluatorAdapter
from mnemos.config import MnemosConfig, ModuleConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.utils.storage import SQLiteStore


def scenario(project="project", operation="remember"):
    source = SourceEvent(
        "original",
        "2026-01-01T00:00:00Z",
        "user",
        "The project uses SQLite.",
        "user_message",
        "explicit" if operation == "remember" else "automatic",
        operation,
        ("gold",),
    )
    replay = replace(
        source,
        event_id="delivery",
        occurred_at="2026-01-02T00:00:00Z",
        lifecycle_operation="replay",
        evidence_fact_ids=(),
        target_source_event_id=source.event_id,
    )
    fact = FactRecord(
        "gold", source.content, False, "project", "synthetic_seed", "forgotten", (source.event_id,)
    )
    session = SessionScenario(
        "session",
        "development",
        "codex",
        project,
        (fact,),
        (),
        (),
        "synthetic_seed",
        "not_run",
        (source, replay),
        (),
    )
    return session, source, replay


@pytest.mark.parametrize("operation", ["remember", "capture"])
def test_replay_dispatches_original_identity_and_payload(operation):
    session, source, replay = scenario(operation=operation)
    calls = []
    service = SimpleNamespace(
        dispatch=lambda op, payload: calls.append((op, payload)) or {"status": "accepted"}
    )
    adapter = OllamaEvaluatorAdapter(lambda *_: service)
    context = adapter._context(session, DEFAULT_PROFILES[0], 1)
    adapter._event_operation(service, source, session, context, DEFAULT_PROFILES[0], 1)
    result = adapter._event_operation(service, replay, session, context, DEFAULT_PROFILES[0], 1)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert result["replayed_source_event_id"] == source.event_id
    assert result["replayed_runtime_event_id"] == calls[0][1]["event_id"]


def test_original_replay_after_forget_and_restart_does_not_resurrect(tmp_path):
    registry = ScopeRegistry(profiles=["alice"])
    project = registry.register_project(tmp_path, user_profile_id="alice", enabled=True)
    config = MnemosConfig(
        modules=ModuleConfig(
            surprisal=False,
            mutable_rag=False,
            affective=False,
            sleep=False,
            spreading=False,
        )
    )
    database = tmp_path / "replay.sqlite"

    def service_factory(*_):
        return LocalMemoryService(
            MnemosEngine(config=config, store=SQLiteStore(str(database))), registry
        )

    session, source, replay = scenario(project.id)
    adapter = OllamaEvaluatorAdapter(service_factory, user_profile_id="alice")
    context = adapter._context(session, DEFAULT_PROFILES[0], 1)
    service = service_factory()
    try:
        adapter._event_operation(service, source, session, context, DEFAULT_PROFILES[0], 1)
        adapter._run_maintenance(service, [], [])
        facts = service.engine.store.get_all()
        assert len(facts) == 1
        adapter._dispatch(
            service,
            "forget",
            {
                "context": context,
                "fact_id": facts[0].id,
                "expected_version": facts[0].version,
            },
        )
        adapter._run_maintenance(service, [], [])
        assert service.engine.store.get_all() == []
    finally:
        service.engine.store.close()

    service = service_factory()
    try:
        adapter._event_operation(service, replay, session, context, DEFAULT_PROFILES[0], 1)
        adapter._run_maintenance(service, [], [])
        assert service.engine.store.get_all() == []
        retrieved = adapter._dispatch(
            service,
            "retrieve",
            {
                "context": context,
                "query": "Which database does the project use?",
                "allowed_scopes": ["project"],
                "top_k": 5,
            },
        )
        assert retrieved["facts"] == []
    finally:
        service.engine.store.close()


def replay_rows():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/local_memory/datasets/local-memory-heldout-v4.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["dataset_version"] = "local-memory-heldout-v6"
    session = rows[1]
    source = next(
        e
        for e in session["events"]
        if e["role"] == "user" and e["lifecycle_operation"] in {"capture", "remember"}
    )
    replay = dict(
        source,
        event_id="replay-delivery",
        occurred_at="2030-01-01T00:00:00Z",
        lifecycle_operation="replay",
        evidence_fact_ids=[],
        target_source_event_id=source["event_id"],
    )
    session["events"].append(replay)
    return rows, replay


def load_rows(tmp_path, rows):
    path = tmp_path / "replay-corpus.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return load_dataset(path)


def test_v6_loads_replay_with_unique_delivery_identity(tmp_path):
    rows, _ = replay_rows()
    dataset = load_rows(tmp_path, rows)
    assert dataset.version == "local-memory-heldout-v6"


@pytest.mark.parametrize("version", [2, 3, 4, 5])
def test_earlier_versions_do_not_silently_gain_replay(tmp_path, version):
    rows, _ = replay_rows()
    rows[0]["dataset_version"] = f"local-memory-heldout-v{version}"
    with pytest.raises(ValueError, match="replay requires"):
        load_rows(tmp_path, rows)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_source_event_id", "absent"),
        ("target_source_event_id", "replay-delivery"),
        ("content", "An altered assertion."),
        ("role", "assistant"),
        ("source_kind", "different-source"),
        ("write_mode", "none"),
        ("target_user_profile_id", "foreign"),
        ("target_project_id", "foreign"),
        ("target_workspace_id", "foreign"),
        ("target_fact_id", "fact"),
        ("replacement_content", "A replacement."),
    ],
)
def test_v6_rejects_changed_or_unresolvable_replay(tmp_path, field, value):
    rows, replay = replay_rows()
    replay[field] = value
    with pytest.raises(ValueError, match="replay"):
        load_rows(tmp_path, rows)


def test_v6_replay_cannot_claim_new_evidence(tmp_path):
    rows, replay = replay_rows()
    replay["evidence_fact_ids"] = [rows[1]["facts"][0]["fact_id"]]
    with pytest.raises(ValueError, match="replay cannot declare"):
        load_rows(tmp_path, rows)


def test_v6_rejects_replay_of_replay(tmp_path):
    rows, replay = replay_rows()
    rows[1]["events"].append(
        dict(
            replay,
            event_id="nested-delivery",
            occurred_at="2031-01-01T00:00:00Z",
            target_source_event_id=replay["event_id"],
        )
    )
    with pytest.raises(ValueError, match="replay source must be a user capture or remember"):
        load_rows(tmp_path, rows)


def test_v6_rejects_future_source(tmp_path):
    rows, replay = replay_rows()
    original = next(
        e for e in rows[1]["events"] if e["event_id"] == replay["target_source_event_id"]
    )
    rows[1]["events"].append(
        dict(original, event_id="future-source", occurred_at="2031-01-01T00:00:00Z")
    )
    replay["target_source_event_id"] = "future-source"
    with pytest.raises(ValueError, match="replay source must exist before"):
        load_rows(tmp_path, rows)
