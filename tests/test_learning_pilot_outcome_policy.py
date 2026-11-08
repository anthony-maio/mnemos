"""Prospective outcome rules; synthetic checks do not pass a native attempt."""

from copy import deepcopy

import pytest

from benchmarks.native_hosts.three_host_contract import require_learning_phase_observations
from tests.test_three_host_contract import TARGET, _forget_calls, call

MISSING = {
    "schema_version": 1,
    "status": "degraded",
    "committed": False,
    "degraded": True,
    "error": "Mnemos service returned HTTP 404 (fact_not_found)",
}


def annotate(calls):
    for index, row in enumerate(calls):
        row.update(
            call_id=f"call-{index}",
            session_id="session-1",
            started_sequence=10 * index + 1,
            completed_sequence=10 * index + 5,
            model_call_correlated=True,
        )
        row.setdefault("dispatch_kind", "mnemos_mcp")
        row.setdefault("outcome", "success")
    return calls


def missing(args=None):
    row = call("mnemos_inspect", {"fact_id": TARGET} if args is None else args, deepcopy(MISSING))
    row["outcome"] = "unsuccessful"
    return row


def discovery(name="list_mcp_resources", args=None):
    key = "resources" if name == "list_mcp_resources" else "resourceTemplates"
    row = call(name, {} if args is None else args, {key: [{"server": "mnemos", "name": "stats"}]})
    row["dispatch_kind"] = "codex_discovery"
    return row


def valid(*, absent=True, discover=True):
    calls = (
        [discovery(), discovery("list_mcp_resource_templates", {"server": "mnemos"})]
        if discover
        else []
    )
    calls.extend(_forget_calls())
    if absent:
        calls.append(missing())
    return annotate(calls)


@pytest.mark.parametrize("absent", [False, True])
@pytest.mark.parametrize("discover", [False, True])
def test_reviewed_deletion_keeps_optional_discovery_and_absence_evidence(absent, discover):
    calls = valid(absent=absent, discover=discover)
    original = deepcopy(calls)
    result = require_learning_phase_observations(calls, phase="forget", fact_id=TARGET)
    assert result["status"] == "phase_policy_satisfied"
    assert len(result["expected_missing_fact_call_ids"]) == int(absent)
    assert len(result["discovery_call_ids"]) == 2 * int(discover)
    assert result["observation_count"] == len(calls)
    assert not result["lifecycle_complete"] and not result["outcomes_rewritten"]
    assert calls == original


@pytest.mark.parametrize("args", [{"chunk_id": TARGET}, {"fact_id": TARGET, "chunk_id": TARGET}])
def test_consistent_inspection_aliases_select_exactly_deleted_fact(args):
    calls = valid(discover=False)
    calls[-1]["args"] = args
    require_learning_phase_observations(calls, phase="forget", fact_id=TARGET)


@pytest.mark.parametrize(
    "defect",
    [
        "wrong-target",
        "conflicting-alias",
        "extra-argument",
        "missing-id",
        "missing-before-delete",
        "overlap-delete",
        "review-overlap",
        "no-delete",
        "no-preview",
        "changed-digest",
        "failed-delete",
        "wrong-delete-receipt",
        "outage",
        "wrong-http-code",
        "wrong-error-code",
        "status-ok",
        "committed",
        "degraded-false",
        "extra-result",
        "boolean-schema",
        "float-schema",
        "integer-committed",
        "duplicate-absence",
        "failed-retrieval",
        "hidden-failure",
        "unknown-outcome",
        "mixed-session",
        "uncorrelated",
        "duplicate-id",
        "reordered",
        "bad-start",
        "bad-end",
        "missing-name",
    ],
)
def test_unbound_absence_or_failed_operation_is_not_accepted(defect):
    calls = valid(discover=False)
    if defect == "wrong-target":
        calls[-1]["args"]["fact_id"] = "other"
    elif defect == "conflicting-alias":
        calls[-1]["args"]["chunk_id"] = "other"
    elif defect == "extra-argument":
        calls[-1]["args"]["scope_id"] = "other"
    elif defect == "missing-id":
        calls[-1]["args"] = {}
    elif defect == "missing-before-delete":
        calls.insert(3, calls.pop())
        annotate(calls)
    elif defect == "overlap-delete":
        calls[-2]["completed_sequence"] = calls[-1]["started_sequence"] + 1
    elif defect == "review-overlap":
        calls[2]["completed_sequence"] = calls[3]["started_sequence"] + 1
    elif defect == "no-delete":
        calls.pop(-2)
    elif defect == "no-preview":
        calls.pop(2)
    elif defect == "changed-digest":
        calls[3]["args"]["preview_digest"] = "b" * 64
    elif defect == "failed-delete":
        calls[3].update(outcome="unsuccessful", result={"status": "rejected"})
    elif defect == "wrong-delete-receipt":
        calls[3]["result"]["fact_id"] = "other"
    elif defect == "outage":
        calls[-1]["result"]["error"] = "connection unavailable"
    elif defect == "wrong-http-code":
        calls[-1]["result"]["error"] = "Mnemos service returned HTTP 500 (fact_not_found)"
    elif defect == "wrong-error-code":
        calls[-1]["result"]["error"] = "Mnemos service returned HTTP 404 (route_not_found)"
    elif defect == "status-ok":
        calls[-1]["result"]["status"] = "ok"
    elif defect == "committed":
        calls[-1]["result"]["committed"] = True
    elif defect == "degraded-false":
        calls[-1]["result"]["degraded"] = False
    elif defect == "extra-result":
        calls[-1]["result"]["fact"] = {"id": TARGET}
    elif defect == "boolean-schema":
        calls[-1]["result"]["schema_version"] = True
    elif defect == "float-schema":
        calls[-1]["result"]["schema_version"] = 1.0
    elif defect == "integer-committed":
        calls[-1]["result"]["committed"] = 0
    elif defect == "duplicate-absence":
        calls.append(missing())
        annotate(calls)
    elif defect == "failed-retrieval":
        calls[-1]["name"] = "mnemos_retrieve"
    elif defect == "hidden-failure":
        calls[-1]["outcome"] = "success"
    elif defect == "unknown-outcome":
        calls[-1]["outcome"] = "expected"
    elif defect == "mixed-session":
        calls[-1]["session_id"] = "other"
    elif defect == "uncorrelated":
        calls[-1]["model_call_correlated"] = False
    elif defect == "duplicate-id":
        calls[-1]["call_id"] = calls[0]["call_id"]
    elif defect == "reordered":
        calls[0], calls[1] = calls[1], calls[0]
    elif defect == "bad-start":
        calls[-1]["started_sequence"] = True
    elif defect == "bad-end":
        calls[-1]["completed_sequence"] = calls[-1]["started_sequence"]
    elif defect == "missing-name":
        calls[-1]["name"] = None
    with pytest.raises(ValueError):
        require_learning_phase_observations(calls, phase="forget", fact_id=TARGET)


@pytest.mark.parametrize(
    "defect",
    [
        "other-server",
        "cursor",
        "duplicate",
        "late",
        "overlap",
        "foreign-result",
        "extra-result",
        "unknown-helper",
        "failed",
        "no-memory",
    ],
)
def test_discovery_cannot_expand_scope_or_hide_calls(defect):
    calls = valid()
    if defect == "other-server":
        calls[0]["args"] = {"server": "other"}
    elif defect == "cursor":
        calls[0]["args"] = {"cursor": "next"}
    elif defect == "duplicate":
        calls.insert(1, discovery())
        annotate(calls)
    elif defect == "late":
        calls.append(calls.pop(0))
        annotate(calls)
    elif defect == "overlap":
        calls[0]["completed_sequence"] = calls[2]["started_sequence"] + 1
    elif defect == "foreign-result":
        calls[0]["result"]["resources"][0]["server"] = "other"
    elif defect == "extra-result":
        calls[0]["result"]["nextCursor"] = "next"
    elif defect == "unknown-helper":
        calls[0]["name"] = "read_mcp_resource"
    elif defect == "failed":
        calls[0].update(outcome="unsuccessful", result={"error": "unavailable"})
    elif defect == "no-memory":
        calls = calls[:2]
    with pytest.raises(ValueError):
        require_learning_phase_observations(calls, phase="forget", fact_id=TARGET)


@pytest.mark.parametrize(
    "phase", ["teach", "recall-v1", "project-2-recall", "correct", "recall-v2", "recall-forgotten"]
)
def test_other_phases_do_not_accept_absence_as_a_successful_operation(phase):
    with pytest.raises(ValueError):
        require_learning_phase_observations(annotate([missing()]), phase=phase)


def test_phase_policy_does_not_replace_recall_or_durable_job_contract():
    calls = annotate([call("mnemos_retrieve", {"query": "test"}, {"facts": []})])
    result = require_learning_phase_observations(calls, phase="recall-v1")
    assert not result["lifecycle_complete"]
    with pytest.raises(ValueError):
        require_learning_phase_observations(calls, phase="unknown")
    with pytest.raises(ValueError):
        require_learning_phase_observations(calls, phase="recall-v1", fact_id=TARGET)
