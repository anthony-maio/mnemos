import time

from benchmarks.local_memory.joint_support_capacity import record_failure


def test_failed_sample_preserves_workload_identity_and_exception():
    result, rows = {}, []
    active = dict(
        phase="measured",
        repeat=3,
        shape="five-token-bound-8-sources",
        start_perf=time.perf_counter(),
    )
    failed = record_failure(result, rows, active, RuntimeError("synthetic OOM"))
    assert result["status"] == "failed"
    assert failed == rows[0]
    assert failed["phase"] == "measured" and failed["repeat"] == 3
    assert failed["shape"] == "five-token-bound-8-sources"
    assert failed["elapsed_until_exception_ms"] >= 0
    assert failed["error"] == "RuntimeError: synthetic OOM"
    assert "start_perf" not in failed


def test_cleanup_error_cannot_retain_passing_status_or_invent_a_sample():
    result = dict(status="necessary_capacity_screen_passed")
    rows = [dict(shape="one-short", elapsed_ms=1)]
    assert record_failure(result, rows, None, RuntimeError("cleanup failed")) is None
    assert result["status"] == "failed"
    assert rows == [dict(shape="one-short", elapsed_ms=1)]
