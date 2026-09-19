import pytest

from benchmarks.native_hosts.hermes_trace import correlate_native_tools, evaluate_phase


def call(call_id, name, args):
    return {"id": call_id, "name": name, "arguments": args}


def result(name, args, payload):
    return {"operation": "tool_result", "name": name, "args": args, "result": payload}


def response(*calls, content=None):
    event = {"operation": "model_response", "tool_calls": list(calls)}
    if content is not None:
        event["content"] = content
    return event


def test_correlates_native_calls_in_result_order_and_parses_json_args():
    records = [
        response(call("r1", "mnemos_retrieve", '{"query":"label"}')),
        result(
            "mnemos_retrieve",
            {"query": "label"},
            {"schema_version": 1, "status": "ok", "facts": []},
        ),
    ]
    assert correlate_native_tools(records) == [
        {
            "call_id": "r1",
            "name": "mnemos_retrieve",
            "args": {"query": "label"},
            "result": {"schema_version": 1, "status": "ok", "facts": []},
        }
    ]


@pytest.mark.parametrize(
    "records",
    [
        [{"operation": "model_response", "content": "remembered it"}],
        [response(call("x", "other_tool", {}))],
        [result("mnemos_forget", {"fact_id": "f", "expected_version": 1}, {"status": "committed"})],
        [
            response(call("x", "mnemos_retrieve", {"query": "a"})),
            result(
                "mnemos_retrieve",
                {"query": "b"},
                {"schema_version": 1, "status": "ok", "facts": []},
            ),
        ],
        [
            response(call("x", "mnemos_retrieve", {"query": "a"})),
            result(
                "mnemos_retrieve",
                {"query": "a"},
                {"schema_version": 1, "status": "degraded", "facts": []},
            ),
        ],
    ],
)
def test_rejects_unproven_or_mismatched_native_trace(records):
    with pytest.raises(ValueError):
        correlate_native_tools(records)


def test_evaluate_phase_requires_structured_lifecycle_evidence():
    records = [
        response(call("t", "mnemos_remember", {"content": "Use cedar."})),
        result(
            "mnemos_remember",
            {"content": "Use cedar."},
            {"schema_version": 1, "status": "committed", "event_id": "e", "job_id": "j"},
        ),
        response(call("j", "mnemos_job_status", {"job_id": "j"})),
        result(
            "mnemos_job_status",
            {"job_id": "j"},
            {"schema_version": 1, "job": {"id": "j", "status": "pending"}},
        ),
    ]
    assert evaluate_phase(
        records, phase="teach", target_id=None, old_content="Use cedar.", new_content=""
    ) == {
        "correlated": True,
        "teach": True,
        "phase_tools_allowed": True,
        "mutations_inspected": True,
    }


def test_evaluate_correct_and_forget_gates():
    records = [
        response(call("i1", "mnemos_inspect", {"fact_id": "f"})),
        result(
            "mnemos_inspect",
            {"fact_id": "f"},
            {"schema_version": 1, "fact": {"id": "f", "version": 1, "content": "old"}},
        ),
        response(
            call("c1", "mnemos_correct", {"fact_id": "f", "content": "new", "expected_version": 1})
        ),
        result(
            "mnemos_correct",
            {"fact_id": "f", "content": "new", "expected_version": 1},
            {"schema_version": 1, "status": "committed", "event_id": "e2", "job_id": "j2"},
        ),
    ]
    assert evaluate_phase(
        records, phase="correct", target_id="f", old_content="old", new_content="new"
    ) == {
        "correlated": True,
        "correct": True,
        "phase_tools_allowed": True,
        "mutations_inspected": True,
    }


def test_evaluate_recall_and_forget_phases_use_exact_fact_shape():
    recall = [
        response(call("r", "mnemos_retrieve", {"query": "label"})),
        result(
            "mnemos_retrieve",
            {"query": "label"},
            {
                "schema_version": 1,
                "status": "ok",
                "facts": [{"id": "f", "version": 2, "content": "new"}],
            },
        ),
    ]
    assert evaluate_phase(
        recall, phase="recall_v2", target_id="f", old_content="old", new_content="new"
    )["recall_v2"]
    forgotten = [
        response(call("i", "mnemos_inspect", {"fact_id": "f"})),
        result(
            "mnemos_inspect",
            {"fact_id": "f"},
            {"schema_version": 1, "fact": {"id": "f", "version": 2, "content": "new"}},
        ),
        response(call("f", "mnemos_forget", {"fact_id": "f", "expected_version": 2})),
        result(
            "mnemos_forget",
            {"fact_id": "f", "expected_version": 2},
            {"schema_version": 1, "status": "committed", "forgotten": True},
        ),
    ]
    assert evaluate_phase(
        forgotten, phase="forget", target_id="f", old_content="old", new_content="new"
    )["forget"]


def test_changed_payload_with_same_call_id_is_rejected():
    records = [
        response(call("same", "mnemos_remember", {"content": "old"})),
        result("mnemos_remember", {"content": "new"}, {"schema_version": 1, "status": "committed"}),
    ]
    with pytest.raises(ValueError):
        correlate_native_tools(records)


@pytest.mark.parametrize(
    "records",
    [
        [
            response(
                call("dup", "mnemos_retrieve", {"query": "a"}),
                call("dup", "mnemos_retrieve", {"query": "b"}),
            ),
        ],
        [response(call("open", "mnemos_retrieve", {"query": "a"}))],
        [
            response(call("one", "mnemos_retrieve", {"query": "a"})),
            result("mnemos_retrieve", {"query": "a"}, {"schema_version": 1, "facts": []}),
            result("mnemos_retrieve", {"query": "a"}, {"schema_version": 1, "facts": []}),
        ],
    ],
)
def test_rejects_duplicate_unmatched_and_replayed_results(records):
    with pytest.raises(ValueError):
        correlate_native_tools(records)


def test_failed_forget_and_post_mutation_inspect_do_not_pass_forget_gate():
    records = [
        response(call("f", "mnemos_forget", {"fact_id": "f", "expected_version": 2})),
        result(
            "mnemos_forget",
            {"fact_id": "f", "expected_version": 2},
            {"schema_version": 1, "status": "committed", "forgotten": False},
        ),
        response(call("i", "mnemos_inspect", {"fact_id": "f"})),
        result(
            "mnemos_inspect",
            {"fact_id": "f"},
            {"schema_version": 1, "fact": {"id": "f", "version": 2, "content": "new"}},
        ),
    ]
    assert not evaluate_phase(
        records, phase="forget", target_id="f", old_content="old", new_content="new"
    )["forget"]


def test_malformed_post_forget_facts_do_not_pass_empty_recall():
    records = [
        response(call("r", "mnemos_retrieve", {"query": "old"})),
        result("mnemos_retrieve", {"query": "old"}, {"schema_version": 1, "facts": [{}]}),
    ]
    assert not evaluate_phase(
        records, phase="recall_forgotten", target_id="f", old_content="old", new_content="new"
    )["recall_forgotten"]


def test_phase_rejects_extra_successful_write_and_uninspected_forget():
    extra_write = [
        response(call("r", "mnemos_retrieve", {"query": "old"})),
        result("mnemos_retrieve", {"query": "old"}, {"schema_version": 1, "facts": []}),
        response(call("w", "mnemos_remember", {"content": "sneak"})),
        result(
            "mnemos_remember",
            {"content": "sneak"},
            {"schema_version": 1, "status": "committed", "event_id": "e", "job_id": "j"},
        ),
    ]
    gates = evaluate_phase(
        extra_write, phase="recall_v1", target_id="f", old_content="old", new_content="new"
    )
    assert gates["phase_tools_allowed"] is False
    uninspected = [
        response(call("f", "mnemos_forget", {"fact_id": "f", "expected_version": 2})),
        result(
            "mnemos_forget",
            {"fact_id": "f", "expected_version": 2},
            {"schema_version": 1, "status": "committed", "forgotten": True},
        ),
    ]
    gates = evaluate_phase(
        uninspected, phase="forget", target_id="f", old_content="old", new_content="new"
    )
    assert gates["mutations_inspected"] is False


def test_null_status_and_empty_fact_content_are_rejected():
    with pytest.raises(ValueError):
        correlate_native_tools(
            [
                response(call("r", "mnemos_retrieve", {"query": "x"})),
                result(
                    "mnemos_retrieve",
                    {"query": "x"},
                    {"schema_version": 1, "status": None, "facts": []},
                ),
            ]
        )
    records = [
        response(call("r", "mnemos_retrieve", {"query": "x"})),
        result(
            "mnemos_retrieve",
            {"query": "x"},
            {"schema_version": 1, "facts": [{"id": "f", "version": 1, "content": ""}]},
        ),
    ]
    assert not evaluate_phase(
        records, phase="recall_forgotten", target_id="f", old_content="old", new_content="new"
    )["recall_forgotten"]


def test_retrieve_accepts_actual_service_shape_without_status():
    records = [
        response(call("r", "mnemos_retrieve", {"query": "label"})),
        result(
            "mnemos_retrieve",
            {"query": "label"},
            {
                "schema_version": 1,
                "facts": [{"id": "f", "version": 1, "content": "old"}],
                "context": "",
            },
        ),
    ]
    assert evaluate_phase(
        records, phase="recall_v1", target_id="f", old_content="old", new_content="new"
    )["recall_v1"]


@pytest.mark.asyncio
async def test_evaluate_accepts_actual_local_memory_retrieve_shape(tmp_path):
    from mnemos.config import MnemosConfig, ModuleConfig
    from mnemos.contracts import RetrievalRequest
    from mnemos.engine import MnemosEngine
    from mnemos.memory_service import LocalMemoryService
    from mnemos.scopes import ScopeRegistry
    from mnemos.utils.embeddings import SimpleEmbeddingProvider
    from mnemos.utils.llm import UnavailableLLMProvider
    from mnemos.utils.storage import SQLiteStore

    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="profile-a", enabled=True)
    context = registry.resolve(tmp_path, user_profile_id="profile-a", session_id="s", host="hermes")
    engine = MnemosEngine(
        config=MnemosConfig(
            modules=ModuleConfig(**{name: False for name in ModuleConfig.model_fields})
        ),
        llm=UnavailableLLMProvider(),
        embedder=SimpleEmbeddingProvider(),
        store=SQLiteStore(str(tmp_path / "memory.sqlite")),
    )
    service = LocalMemoryService(engine, registry)
    try:
        remembered = await service.dispatch(
            "remember",
            {"context": context.model_dump(), "content": "Actual shape fact.", "role": "user"},
        )
        await service.maintenance.run_once(force=True)
        actual = await service.retrieve(
            RetrievalRequest(
                query="shape fact",
                context=context,
                allowed_scopes=["project"],
                top_k=5,
                max_context_tokens=1200,
            )
        )
        fact = actual["facts"][0]
        inspected = await service.dispatch(
            "inspect", {"context": context.model_dump(), "fact_id": fact["id"]}
        )
        job_status = await service.dispatch(
            "job", {"context": context.model_dump(), "job_id": remembered["job_id"]}
        )
        assert inspected["schema_version"] == 1
        assert inspected["fact"]["id"] == fact["id"]
        assert job_status["schema_version"] == 1
        assert job_status["job"]["id"] == remembered["job_id"]
        assert job_status["job"]["status"] in {"pending", "leased", "completed"}
        records = [
            response(call("r", "mnemos_retrieve", {"query": "shape fact"})),
            result("mnemos_retrieve", {"query": "shape fact"}, actual),
        ]
        assert evaluate_phase(
            records,
            phase="recall_v1",
            target_id=fact["id"],
            old_content=fact["content"],
            new_content="",
        )["recall_v1"]
        assert remembered["status"] == "committed"
    finally:
        engine.store.close()
