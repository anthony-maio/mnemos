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


def test_related_forget_requires_exact_reviewed_selection_and_actual_receipt():
    inspected = call(
        "mnemos_inspect", {"chunk_id": "target"}, {"fact": {"id": "target", "version": 2}}
    )
    preview = call(
        "mnemos_forget_preview",
        {"fact_id": "target", "expected_version": 2},
        {
            "complete": True,
            "preview_digest": "digest",
            "candidates": [{"fact_id": "copy"}],
            "source_events": [{"source_event_id": "source"}],
        },
    )
    forgotten = call(
        "mnemos_forget_related",
        {
            "fact_id": "target",
            "expected_version": 2,
            "preview_digest": "digest",
            "selected_fact_ids": ["copy"],
            "selected_source_event_ids": ["source"],
        },
        {
            "status": "committed",
            "forgotten": True,
            "fact_id": "target",
            "selected_fact_ids": ["target", "copy"],
            "selected_source_event_ids": ["source"],
        },
    )
    require_forget([inspected, preview, forgotten], "target")
    wrong = copy.deepcopy(forgotten)
    wrong["args"]["selected_fact_ids"] = ["never-reviewed"]
    with pytest.raises(ValueError, match="not in the reviewed"):
        require_forget([inspected, preview, wrong], "target")
    wrong = copy.deepcopy(forgotten)
    wrong["result"]["selected_fact_ids"] = ["target"]
    with pytest.raises(ValueError, match="differs"):
        require_forget([inspected, preview, wrong], "target")
