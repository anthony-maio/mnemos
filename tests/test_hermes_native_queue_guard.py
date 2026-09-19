"""The native lifecycle runner must not overlap a still-active model queue."""

import json

import pytest

from benchmarks.native_hosts.hermes_lifecycle_run import (
    controls_unchanged,
    require_prior_queue_terminal,
    verify_committed_inputs,
)


@pytest.mark.parametrize("status", ["finished", "finished_with_incomplete", "failed"])
def test_accepts_actual_coordinator_terminal_status_only_after_process_exit(tmp_path, status):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"status": status, "pid": 12}))
    checked = []

    def probe(pid):
        checked.append(pid)
        return False

    result = require_prior_queue_terminal(path, [12, 13, 14], probe=probe)
    assert checked == [12, 13, 14]
    assert result["state"]["status"] == status
    assert result["excluded_pids"] == [12, 13, 14]


@pytest.mark.parametrize(
    "status", ["running", "waiting_for_endpoint", "stage_completed_incomplete"]
)
def test_nonterminal_state_never_reaches_process_probe(tmp_path, status):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"status": status, "pid": 12}))

    def probe(pid):
        pytest.fail("A nonterminal queue must be rejected before any further work")

    with pytest.raises(ValueError, match="not terminal"):
        require_prior_queue_terminal(path, [12], probe=probe)


def test_terminal_record_does_not_override_a_live_worker(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"status": "finished_with_incomplete", "pid": 12}))
    with pytest.raises(ValueError, match="still live"):
        require_prior_queue_terminal(path, [12, 13], probe=lambda pid: pid == 13)


def test_requires_recorded_coordinator_and_valid_process_ids(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"status": "finished", "pid": 12}))
    for pids in ([], [13], [12, True], [12, -1]):
        with pytest.raises(ValueError, match="process IDs"):
            require_prior_queue_terminal(path, pids, probe=lambda pid: False)


def test_control_check_rejects_duplicate_content_and_altered_evidence():
    import copy

    before = {
        "facts": [{"id": "control", "content": "keep", "version": 1, "metadata": "{}"}],
        "evidence": [{"fact_id": "control", "source_event_id": "source", "excerpt": "keep"}],
        "events": [{"id": "source", "content": "keep"}],
    }
    after = copy.deepcopy(before)
    after["facts"].append({"id": "target", "content": "other", "version": 2, "metadata": "{}"})
    assert controls_unchanged(before, after)
    for table, row in (
        ("facts", {"id": "duplicate", "content": "keep", "version": 1, "metadata": "{}"}),
        ("evidence", {"fact_id": "control", "source_event_id": "extra", "excerpt": "keep"}),
    ):
        changed = copy.deepcopy(after)
        changed[table].append(row)
        assert not controls_unchanged(before, changed)
    changed = copy.deepcopy(after)
    changed["events"][0]["content"] = "changed source"
    assert not controls_unchanged(before, changed)


def test_model_run_rejects_uncommitted_or_changed_helper(tmp_path, monkeypatch):
    import subprocess

    helper = tmp_path / "helper.py"
    helper.write_text("committed\n")
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: b"committed\n")
    verify_committed_inputs(tmp_path, ["helper.py"], "revision")
    helper.write_text("changed\n")
    with pytest.raises(ValueError, match="committed"):
        verify_committed_inputs(tmp_path, ["helper.py"], "revision")

    def untracked(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])

    monkeypatch.setattr(subprocess, "check_output", untracked)
    with pytest.raises(ValueError, match="committed"):
        verify_committed_inputs(tmp_path, ["helper.py"], "revision")
