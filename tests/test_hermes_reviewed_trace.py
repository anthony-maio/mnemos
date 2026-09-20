"""Native reviewed deletion needs observed previews, inspections, and exact results."""

from copy import deepcopy

import pytest

from benchmarks.native_hosts import hermes_trace_v2 as trace
from tests.test_hermes_native_trace import call, response, result


def reviewed_trace():
    preview_args = {"fact_id": "target", "expected_version": 2}
    apply_args = {
        **preview_args,
        "preview_digest": "a" * 64,
        "preview_limit": 100,
        "selected_fact_ids": ["copy"],
        "selected_source_event_ids": ["source"],
    }
    preview = {
        "schema_version": 1,
        "status": "preview",
        "target": {"fact_id": "target", "version": 2, "content": "new"},
        "complete": True,
        "limit": 100,
        "preview_digest": "a" * 64,
        "candidates": [
            {
                "fact_id": "copy",
                "version": 1,
                "content": "Captured old",
                "required_source_event_ids": ["source"],
                "blocked_source_event_ids": [],
            }
        ],
        "source_events": [
            {"source_event_id": "source", "selection_allowed": True, "derived_fact_ids": ["copy"]}
        ],
    }
    return [
        response(call("inspect-target", "mnemos_inspect", {"fact_id": "target"})),
        result(
            "mnemos_inspect",
            {"fact_id": "target"},
            {"schema_version": 1, "fact": {"id": "target", "version": 2, "content": "new"}},
        ),
        response(call("preview", "mnemos_forget_preview", preview_args)),
        result("mnemos_forget_preview", preview_args, preview),
        response(call("inspect-copy", "mnemos_inspect", {"fact_id": "copy"})),
        result(
            "mnemos_inspect",
            {"fact_id": "copy"},
            {"schema_version": 1, "fact": {"id": "copy", "version": 1, "content": "Captured old"}},
        ),
        response(call("apply", "mnemos_forget_related", apply_args)),
        result(
            "mnemos_forget_related",
            apply_args,
            {
                "schema_version": 1,
                "status": "committed",
                "forgotten": True,
                "fact_id": "target",
                "selected_fact_ids": ["target", "copy"],
                "selected_source_event_ids": ["source"],
            },
        ),
    ]


def evaluate(records, phase="forget"):
    return trace.evaluate_phase(
        records, phase=phase, target_id="target", old_content="old", new_content="new"
    )


def test_accepts_model_reviewed_and_inspected_deletion():
    gates = evaluate(reviewed_trace())
    assert gates["forget"] is True
    assert gates["reviewed_forget_verified"] is True
    assert all(gates.values())


@pytest.mark.parametrize(
    "change",
    [
        "incomplete",
        "digest",
        "limit",
        "unknown_fact",
        "missing_source",
        "extra_result_fact",
        "extra_result_source",
        "wrong_version",
        "missing_copy_inspect",
        "missing_target_inspect",
        "source_disallowed",
        "unselected_source_dependency",
        "uncommitted",
        "duplicate_selected",
    ],
)
def test_rejects_unproven_reviewed_deletion(change):
    records = reviewed_trace()
    preview = records[3]["result"]
    args = records[6]["tool_calls"][0]["arguments"]
    applied = records[7]["result"]
    if change == "incomplete":
        preview["complete"] = False
    elif change == "digest":
        args["preview_digest"] = "b" * 64
    elif change == "limit":
        args["preview_limit"] = 200
    elif change == "unknown_fact":
        args["selected_fact_ids"] = ["unlisted"]
    elif change == "missing_source":
        args["selected_source_event_ids"] = []
    elif change == "extra_result_fact":
        applied["selected_fact_ids"].append("unrelated")
    elif change == "extra_result_source":
        applied["selected_source_event_ids"].append("unrelated")
    elif change == "wrong_version":
        records[5]["result"]["fact"]["version"] = 2
    elif change == "missing_copy_inspect":
        del records[4:6]
    elif change == "missing_target_inspect":
        del records[:2]
    elif change == "source_disallowed":
        preview["source_events"][0]["selection_allowed"] = False
    elif change == "unselected_source_dependency":
        preview["source_events"][0]["derived_fact_ids"].append("unrelated")
    elif change == "uncommitted":
        applied["status"] = "pending"
    elif change == "duplicate_selected":
        args["selected_fact_ids"].append("copy")
    assert evaluate(records)["reviewed_forget_verified"] is False


def test_preview_and_apply_in_one_model_response_is_not_observed_review():
    records = reviewed_trace()
    apply_call = records.pop(6)["tool_calls"][0]
    records[2]["tool_calls"].append(apply_call)
    assert evaluate(records)["reviewed_forget_verified"] is False


def test_reviewed_tools_are_not_allowed_in_recall_phase():
    assert evaluate(reviewed_trace(), phase="recall_v2")["phase_tools_allowed"] is False


def test_direct_forget_does_not_prove_the_new_reviewed_workflow():
    records = reviewed_trace()[:2]
    args = {"fact_id": "target", "expected_version": 2}
    records.extend(
        [
            response(call("direct", "mnemos_forget", args)),
            result(
                "mnemos_forget",
                args,
                {"schema_version": 1, "status": "committed", "forgotten": True},
            ),
        ]
    )
    assert evaluate(records)["reviewed_forget_verified"] is False


def test_original_evaluator_still_rejects_new_tools():
    from benchmarks.native_hosts.hermes_trace import correlate_native_tools

    with pytest.raises(ValueError, match="unsupported native tool"):
        correlate_native_tools(deepcopy(reviewed_trace()))


@pytest.mark.asyncio
async def test_reviewed_evaluator_accepts_actual_service_result_and_database_state(tmp_path):
    import json

    from benchmarks.native_hosts.hermes_lifecycle_run_v5 import read_rows
    from mnemos.adapters.hermes_tools import dispatch_tool
    from tests.test_hermes_related_forget import LocalServiceClient
    from tests.test_related_forget import _seed_overlap, _service

    memory, context, _, _ = _service(tmp_path)
    client = LocalServiceClient(memory)
    records = []

    async def invoke(name, args):
        records.append(response(call(f"call-{len(records)}", name, deepcopy(args))))
        payload = json.loads(await dispatch_tool(client, context, name, args))
        records.append(result(name, deepcopy(args), payload))
        return payload

    try:
        old, target, copy = await _seed_overlap(memory, context)
        await memory.dispatch(
            "correct",
            {
                "context": context.model_dump(mode="json"),
                "event_id": "revision",
                "fact_id": target.id,
                "expected_version": 1,
                "content": "new",
            },
        )
        await memory.dispatch("maintenance", {})
        await invoke("mnemos_inspect", {"fact_id": target.id})
        preview = await invoke(
            "mnemos_forget_preview", {"fact_id": target.id, "expected_version": 2}
        )
        await invoke("mnemos_inspect", {"fact_id": copy.id})
        assert not trace.no_retained_labels(
            read_rows(tmp_path / "memory.sqlite"),
            context=context.model_dump(mode="json"),
            labels=["related-forget-old"],
        )
        await invoke(
            "mnemos_forget_related",
            {
                "fact_id": target.id,
                "expected_version": 2,
                "preview_digest": preview["preview_digest"],
                "preview_limit": preview["limit"],
                "selected_fact_ids": [copy.id],
                "selected_source_event_ids": [
                    row["source_event_id"] for row in preview["source_events"]
                ],
            },
        )
        gates = trace.evaluate_phase(
            records, phase="forget", target_id=target.id, old_content=old, new_content="new"
        )
        assert all(gates.values()), gates
        state = read_rows(tmp_path / "memory.sqlite")
        assert trace.no_retained_labels(
            state, context=context.model_dump(mode="json"), labels=["related-forget-old"]
        )
        assert trace.reviewed_deletions_persisted(records, state)
        state["facts"].append({"id": copy.id})
        assert not trace.reviewed_deletions_persisted(records, state)
    finally:
        memory.engine.store.close()


@pytest.mark.parametrize("location", ["facts", "events", "evidence", "revisions", "jobs"])
def test_scoped_label_scan_detects_retained_copies_outside_retrieval(location):
    import json

    metadata = {"scope": "project", "scope_id": "project-a", "user_profile_id": "profile-a"}
    context = {"project_id": "project-a", "user_profile_id": "profile-a"}
    state = {name: [] for name in ("facts", "events", "evidence", "revisions", "jobs")}
    state["facts"] = [{"id": "retained", "content": "unrelated", "metadata": json.dumps(metadata)}]
    marker = "maple-ridge-842"
    if location == "facts":
        state[location][0]["content"] = marker
    elif location == "events":
        state[location] = [{"id": "source", "content": marker, "metadata": json.dumps(metadata)}]
    elif location in {"evidence", "revisions"}:
        state[location] = [{"fact_id": "retained", "excerpt": marker}]
    else:
        state[location] = [
            {"id": "job", "payload": json.dumps({"context": context, "content": marker})}
        ]
    assert not trace.no_retained_labels(state, context=context, labels=[marker])
    foreign = {**context, "project_id": "foreign-project"}
    assert trace.no_retained_labels(state, context=foreign, labels=[marker])


def test_persisted_deletion_requires_source_and_tombstone_evidence():
    records = reviewed_trace()
    state = {name: [] for name in ("facts", "events", "evidence", "revisions", "suppressions")}
    state["suppressions"] = [
        {"fact_id": "target"},
        {"fact_id": "copy"},
        {"source_event_id": "source"},
    ]
    assert trace.reviewed_deletions_persisted(records, state)
    for table, leftover in [
        ("events", {"id": "source"}),
        ("evidence", {"fact_id": "copy"}),
        ("revisions", {"fact_id": "target"}),
    ]:
        mutated = deepcopy(state)
        mutated[table].append(leftover)
        assert not trace.reviewed_deletions_persisted(records, mutated)
    state["suppressions"].pop()
    assert not trace.reviewed_deletions_persisted(records, state)


@pytest.mark.parametrize("table", ["evidence", "revisions"])
def test_scoped_label_scan_keeps_attribution_after_fact_deletion(table):
    state = {name: [] for name in ("facts", "events", "evidence", "revisions", "jobs")}
    state[table] = [{"fact_id": "deleted-project-fact", "content": "maple-ridge-842"}]
    assert not trace.no_retained_labels(
        state,
        context={"project_id": "p", "user_profile_id": "u"},
        labels=["maple-ridge-842"],
        known_fact_ids={"deleted-project-fact"},
    )


def test_persisted_marker_scan_does_not_match_different_identifier_with_shared_prefix():
    state = {name: [] for name in ("facts", "events", "evidence", "revisions", "jobs")}
    state["facts"] = [
        {
            "id": "control",
            "content": "maple-ridge-842-archive",
            "metadata": {"scope": "project", "scope_id": "p", "user_profile_id": "u"},
        }
    ]
    context = {"project_id": "p", "user_profile_id": "u"}
    assert trace.no_retained_labels(state, context=context, labels=["maple-ridge-842"])
    state["facts"][0]["content"] = "Label: MAPLE-RIDGE-842."
    assert not trace.no_retained_labels(state, context=context, labels=["maple-ridge-842"])


def test_non_forget_gates_match_original_on_all_archived_v5_phases():
    import json
    from pathlib import Path

    from benchmarks.native_hosts import hermes_trace as original

    archive = (
        Path(__file__).resolve().parents[1] / "docs/validation/2026-09-19/hermes-native-cohort-v5"
    )
    checked = 0
    for attempt in sorted(archive.glob("attempt-*")):
        if not attempt.is_dir():
            continue
        report = json.loads((attempt / "report.json").read_text())
        for phase in report["phases"]:
            name = phase["phase"]
            if name in {"forget", "source-replay"}:
                continue
            records = [
                json.loads(line)
                for line in (attempt / name / "trace.jsonl").read_text().splitlines()
            ]
            args = {
                "phase": name,
                "target_id": report.get("target_id"),
                "old_content": "The project release label is cedar-ridge-731.",
                "new_content": "The project release label is maple-ridge-842.",
            }
            assert trace.evaluate_phase(records, **args) == original.evaluate_phase(records, **args)
            checked += 1
    assert checked == 14
