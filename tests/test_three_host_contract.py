import copy

import pytest

from benchmarks.native_hosts.three_host_contract import (
    require_absent,
    require_correct,
    require_job_completion,
    require_native_calls,
    require_recalled,
    require_teach,
    require_forget,
)


def call(name, args, result):
    return {"name": name, "args": args, "result": result, "model_call_correlated": True}


def test_teaching_requires_the_real_job_to_complete_after_write():
    write = call(
        "mnemos_store",
        {"content": "fact"},
        {"status": "committed", "event_id": "source", "job_id": "job-1"},
    )
    completed = call(
        "mnemos_job_status", {"job_id": "job-1"}, {"job": {"id": "job-1", "status": "completed"}}
    )
    assert require_teach([write, completed], "fact") == write["result"]
    for bad in [[], [completed, write], [write, write, completed]]:
        with pytest.raises(ValueError):
            require_teach(bad, "fact")
    for field, value in [("id", "different-job"), ("status", "pending")]:
        altered = copy.deepcopy(completed)
        altered["result"]["job"][field] = value
        with pytest.raises(ValueError, match="actual job"):
            require_job_completion([altered], write["result"])


def test_uncorrelated_or_extra_tool_cannot_pass_native_acceptance():
    row = call("mnemos_retrieve", {}, {})
    row["model_call_correlated"] = False
    with pytest.raises(ValueError, match="correlation"):
        require_native_calls([row], {"mnemos_retrieve"})
    with pytest.raises(ValueError, match="unapproved"):
        require_native_calls([call("mnemos_store", {}, {})], {"mnemos_retrieve"})


def test_recall_requires_scope_identity_version_assertion_and_independent_query():
    fact = {
        "id": "original",
        "content": "new",
        "version": 2,
        "scope": "project",
        "scope_id": "project-a",
        "assertion_status": "asserted",
    }
    row = call("mnemos_retrieve", {"query": "release label"}, {"facts": [fact]})
    expected = dict(
        fact_id="original", content="new", version=2, scope_id="project-a", query="release label"
    )
    require_recalled([row], **expected)
    for field, value in [
        ("id", "duplicate"),
        ("version", 1),
        ("scope_id", "foreign"),
        ("assertion_status", "captured"),
    ]:
        altered = copy.deepcopy(row)
        altered["result"]["facts"][0][field] = value
        with pytest.raises(ValueError, match="exact scoped"):
            require_recalled([altered], **expected)
    with pytest.raises(ValueError):
        require_recalled([row], **{**expected, "query": "review label"})


def test_correction_requires_prior_inspection_and_completed_actual_job():
    inspect = call(
        "mnemos_inspect", {"fact_id": "target"}, {"fact": {"id": "target", "version": 1}}
    )
    edit = call(
        "mnemos_correct",
        {"fact_id": "target", "expected_version": 1, "content": "new"},
        {"status": "committed", "job_id": "job"},
    )
    done = call(
        "mnemos_job_status", {"job_id": "job"}, {"job": {"id": "job", "status": "completed"}}
    )
    require_correct([inspect, edit, done], "target", "new")
    with pytest.raises(ValueError, match="prior inspection"):
        require_correct([edit, inspect, done], "target", "new")


def test_forget_checks_raw_sources_and_revisions_not_only_fact_absence():
    state = {
        "facts": [],
        "events": [],
        "revisions": [],
        "evidence": [],
        "jobs": [],
        "edges": [],
        "suppressions": [{"fact_id": "target", "canonical_key": "private-label"}],
    }
    require_absent(state, "target", ["private-label"])
    for table in ("facts", "events", "revisions", "evidence", "jobs"):
        changed = copy.deepcopy(state)
        changed[table] = [{"id": "copy", "content": "private-label"}]
        with pytest.raises(ValueError, match="remains"):
            require_absent(changed, "target", ["private-label"])
    state["edges"] = [{"source_id": "target", "target_id": "removed-copy"}]
    with pytest.raises(ValueError, match="Derived graph"):
        require_absent(state, "target", ["private-label"])


TARGET = "fact-target"
EXTRA = "fact-captured-copy"
SOURCE = "event-captured-copy"
TARGET_CONTENT = "The project release label is maple-ridge-842."
DIGEST = "a" * 64


def _forget_calls() -> list[dict]:
    return [
        call(
            "mnemos_inspect",
            {"fact_id": TARGET},
            {
                "schema_version": 1,
                "fact": {"id": TARGET, "version": 2, "content": TARGET_CONTENT},
            },
        ),
        call(
            "mnemos_inspect",
            {"fact_id": EXTRA},
            {
                "schema_version": 1,
                "fact": {"id": EXTRA, "version": 1, "content": TARGET_CONTENT},
            },
        ),
        call(
            "mnemos_forget_preview",
            {"fact_id": TARGET, "expected_version": 2, "limit": 100},
            {
                "schema_version": 1,
                "status": "preview",
                "complete": True,
                "preview_digest": DIGEST,
                "limit": 100,
                "target": {
                    "fact_id": TARGET,
                    "version": 2,
                    "content": TARGET_CONTENT,
                },
                "candidates": [
                    {
                        "fact_id": TARGET,
                        "version": 2,
                        "content": TARGET_CONTENT,
                        "required_source_event_ids": [],
                        "blocked_source_event_ids": [],
                    },
                    {
                        "fact_id": EXTRA,
                        "version": 1,
                        "content": TARGET_CONTENT,
                        "required_source_event_ids": [SOURCE],
                        "blocked_source_event_ids": [],
                    },
                ],
                "source_events": [
                    {
                        "source_event_id": SOURCE,
                        "selection_allowed": True,
                        "derived_fact_ids": [EXTRA, TARGET],
                    }
                ],
            },
        ),
        call(
            "mnemos_forget_related",
            {
                "fact_id": TARGET,
                "expected_version": 2,
                "preview_digest": DIGEST,
                "preview_limit": 100,
                "selected_fact_ids": [EXTRA],
                "selected_source_event_ids": [SOURCE],
            },
            {
                "schema_version": 1,
                "status": "committed",
                "forgotten": True,
                "fact_id": TARGET,
                "selected_fact_ids": [TARGET, EXTRA],
                "selected_source_event_ids": [SOURCE],
            },
        ),
    ]


def test_related_forget_requires_exact_reviewed_selection_and_actual_receipt():
    calls = _forget_calls()
    require_forget(calls, TARGET)
    wrong = copy.deepcopy(calls)
    wrong[3]["args"]["selected_fact_ids"] = ["never-reviewed"]
    with pytest.raises(ValueError, match="not in the reviewed"):
        require_forget(wrong, TARGET)
    wrong = copy.deepcopy(calls)
    wrong[3]["result"]["selected_fact_ids"] = [TARGET]
    with pytest.raises(ValueError, match="differs"):
        require_forget(wrong, TARGET)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda calls: calls[3]["args"].update(preview_limit=99),
        lambda calls: calls[2]["result"]["source_events"][0].update(selection_allowed=False),
        lambda calls: calls[2]["result"]["candidates"][1].update(
            required_source_event_ids=["missing-event"]
        ),
        lambda calls: calls[1]["result"]["fact"].update(content="stale copy"),
    ],
)
def test_related_forget_rejects_unbound_review_state(mutate) -> None:
    calls = _forget_calls()
    mutate(calls)
    with pytest.raises(ValueError, match="Forget"):
        require_forget(calls, TARGET)


def test_related_forget_rejects_missing_target_inspection() -> None:
    calls = _forget_calls()
    calls.pop(0)
    with pytest.raises(ValueError, match="Forget"):
        require_forget(calls, TARGET)


def test_related_forget_rejects_missing_selected_candidate_inspection() -> None:
    calls = _forget_calls()
    calls.pop(1)
    with pytest.raises(ValueError, match="Forget"):
        require_forget(calls, TARGET)


def test_related_forget_rejects_result_selection_mismatch() -> None:
    calls = _forget_calls()
    calls[3]["result"]["selected_fact_ids"] = [TARGET]
    with pytest.raises(ValueError, match="Forget"):
        require_forget(calls, TARGET)


def test_related_forget_rejects_unknown_apply_schema() -> None:
    calls = _forget_calls()
    calls[3]["result"]["schema_version"] = 2
    with pytest.raises(ValueError, match="apply.*schema"):
        require_forget(calls, TARGET)


def test_related_forget_rejects_unselected_source_closure() -> None:
    calls = _forget_calls()
    calls[2]["result"]["source_events"][0]["derived_fact_ids"] = [EXTRA, "unselected"]
    with pytest.raises(ValueError, match="closure"):
        require_forget(calls, TARGET)


def test_related_forget_rejects_duplicate_selection_ids() -> None:
    calls = _forget_calls()
    calls[3]["args"]["selected_fact_ids"] = [EXTRA, EXTRA]
    with pytest.raises(ValueError, match="duplicates"):
        require_forget(calls, TARGET)


def test_related_forget_rejects_inconsistent_target_content() -> None:
    calls = _forget_calls()
    calls[2]["result"]["target"]["content"] = "different target content"
    with pytest.raises(ValueError, match="Forget"):
        require_forget(calls, TARGET)


def test_related_forget_rejects_non_integer_target_version() -> None:
    calls = _forget_calls()
    calls[2]["result"]["target"]["version"] = 2.0
    with pytest.raises(ValueError, match="Forget"):
        require_forget(calls, TARGET)
