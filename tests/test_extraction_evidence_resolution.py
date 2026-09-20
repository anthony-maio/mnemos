from types import SimpleNamespace

import pytest

from benchmarks.local_memory.contracts import FactRecord, SessionScenario, SourceEvent
from benchmarks.local_memory.ollama_adapter import OllamaEvaluatorAdapter


@pytest.mark.parametrize(
    ("extra_source", "expected"),
    [(None, True), ("missing", False), ("foreign", False), ("unlabeled", False)],
)
def test_every_claimed_extraction_source_must_resolve(extra_source, expected):
    fact = FactRecord(
        "fact", "Use SQLite locally.", True, "project", "synthetic_seed", "supported", ("source",)
    )
    event = SourceEvent(
        "source",
        "2026-01-01T00:00:00Z",
        "user",
        fact.content,
        "user_message",
        "automatic",
        "capture",
        (fact.fact_id,),
    )
    session = SessionScenario(
        "session",
        "development",
        "codex",
        "project",
        (fact,),
        (),
        (),
        "synthetic_seed",
        "not_run",
        (event,),
        (),
    )
    sources = ["valid"] + ([extra_source] if extra_source else [])
    events = {
        "valid": SimpleNamespace(metadata={"source_event_id": "source"}),
        "foreign": SimpleNamespace(metadata={"source_event_id": "other-session-source"}),
        "unlabeled": SimpleNamespace(metadata={}),
    }
    chunk = SimpleNamespace(
        id="runtime-fact",
        content=fact.content,
        metadata={
            "source_event_ids": sources,
            "scope": "project",
            "scope_id": "project",
            "user_profile_id": "t18-synthetic-profile",
        },
    )
    store = SimpleNamespace(get_all=lambda: [chunk], get_episodic_event=events.get)
    service = SimpleNamespace(engine=SimpleNamespace(store=store))
    adapter = OllamaEvaluatorAdapter(service_factory=lambda **_: service)
    values, details = adapter._extraction_judgements(service, session)
    assert values == [expected]
    assert details[0]["accepted"] is expected
    assert details[0]["source_resolution_complete"] is expected
