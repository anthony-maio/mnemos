import json
import time

import pytest

from benchmarks.local_memory import reduced_depth_capacity as probe


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit, RuntimeError])
def test_terminal_failure_writes_result_and_prevents_restart(tmp_path, monkeypatch, error):
    (tmp_path / "freeze.json").write_text("{}")

    def interrupt(_):
        raise error("injected interruption")

    monkeypatch.setattr(probe, "digest", interrupt)
    with pytest.raises(error):
        probe.run(tmp_path)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "failed"
    assert error.__name__ in result["error"]
    with pytest.raises(FileExistsError):
        probe.run(tmp_path)


def test_interrupt_retains_active_sample_and_resets_pass_status():
    result = dict(status="necessary_capacity_screen_passed")
    rows = []
    active = dict(
        phase="measured",
        repeat=9,
        shape="five-token-bound-1-sources",
        start_perf=time.perf_counter(),
    )
    row = probe.record_failure(result, rows, active, KeyboardInterrupt())
    assert result["status"] == "failed"
    assert row["repeat"] == 9 and row["shape"] == active["shape"]
    assert row["status"] == "failed" and row["elapsed_until_exception_ms"] >= 0

