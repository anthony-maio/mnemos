import json
import os
import subprocess
import sys
import runpy
from pathlib import Path
import uuid
from unittest.mock import patch

import pytest

from benchmarks.native_hosts.windows_owned_job import run_owned_job

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows process containment")
if os.name == "nt":
    pytest.importorskip("win32job", reason="Optional pywin32 native-validation dependency")


def test_normal_command_is_contained_and_exits(tmp_path):
    result = run_owned_job(
        [sys.executable, "-I", "-S", "-c", "print('owned probe')"],
        cwd=tmp_path,
        env=dict(os.environ),
        output=tmp_path / "out",
        timeout=10,
    )
    assert result["status"] == "passed" and result["active_after_cleanup"] == 0
    assert result["total_job_processes"] >= 2
    assert b"owned probe" in (tmp_path / "out/stdout.log").read_bytes()


def test_timeout_cleans_broker_child_and_grandchild(tmp_path):
    script = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-I','-S','-c','import time;time.sleep(60)']); time.sleep(60)"
    with pytest.raises(subprocess.TimeoutExpired):
        run_owned_job(
            [sys.executable, "-I", "-S", "-c", script],
            cwd=tmp_path,
            env=dict(os.environ),
            output=tmp_path / "out",
            timeout=2,
        )
    result = json.loads((tmp_path / "out/process.json").read_text())
    assert result["status"] == "failed" and result["active_after_cleanup"] == 0
    assert result["total_job_processes"] >= 3


def test_assignment_failure_never_releases_target_command(tmp_path):
    import win32job

    marker = tmp_path / "should-not-exist"
    script = "from pathlib import Path;Path(" + repr(str(marker)) + ").touch()"
    with patch.object(win32job, "AssignProcessToJobObject", side_effect=RuntimeError("injected")):
        with pytest.raises(RuntimeError, match="injected"):
            run_owned_job(
                [sys.executable, "-I", "-S", "-c", script],
                cwd=tmp_path,
                env=dict(os.environ),
                output=tmp_path / "out",
                timeout=10,
            )
    assert not marker.exists()
    result = json.loads((tmp_path / "out/process.json").read_text())
    assert not result["input_delivery_started"] and result["active_after_cleanup"] == 0


def test_parent_exit_still_cleans_surviving_descendant(tmp_path):
    script = "import subprocess,sys; subprocess.Popen([sys.executable,'-I','-S','-c','import time;time.sleep(60)'])"
    result = run_owned_job(
        [sys.executable, "-I", "-S", "-c", script],
        cwd=tmp_path,
        env=dict(os.environ),
        output=tmp_path / "out",
        timeout=10,
    )
    assert result["status"] == "passed"
    assert result["active_before_cleanup"] >= 1
    assert result["active_after_cleanup"] == 0
    assert result["total_job_processes"] >= 3


def test_timeout_includes_blocked_input_delivery(tmp_path):
    import time

    env = dict(os.environ, OWNED_JOB_TEST_PADDING="x" * 1_000_000)
    start = time.monotonic()
    with patch(
        "benchmarks.native_hosts.windows_owned_job.BROKER",
        "import time; time.sleep(60)",
    ):
        with pytest.raises(subprocess.TimeoutExpired):
            run_owned_job(
                [sys.executable, "-c", "pass"],
                cwd=tmp_path,
                env=env,
                output=tmp_path / "out",
                timeout=2,
            )
    assert time.monotonic() - start < 15
    result = json.loads((tmp_path / "out/process.json").read_text())
    assert result["input_delivery_started"]
    assert result["active_after_cleanup"] == 0


def test_named_supervisor_membership_is_verified(tmp_path):
    name = "Local\\MnemosReplayTest-" + uuid.uuid4().hex
    helper = (
        Path(__file__).resolve().parents[1] / "benchmarks/native_hosts/replay_ack_comparison.py"
    )
    script = f"import runpy;runpy.run_path({str(helper)!r})['require_job']({name!r})"
    result = run_owned_job(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        env=dict(os.environ),
        output=tmp_path / "out",
        timeout=15,
        job_name=name,
    )
    assert result["status"] == "passed" and result["job_name"] == name
    assert result["active_after_cleanup"] == 0


def test_unrelated_job_is_rejected_and_never_reconfigured(tmp_path):
    import win32job

    name = "Local\\MnemosReplayTest-" + uuid.uuid4().hex
    job = win32job.CreateJobObject(None, name)
    helper = (
        Path(__file__).resolve().parents[1] / "benchmarks/native_hosts/replay_ack_comparison.py"
    )
    try:
        with pytest.raises(RuntimeError, match="outside"):
            runpy.run_path(str(helper))["require_job"](name)
        with pytest.raises(RuntimeError, match="already exists"):
            run_owned_job(
                [sys.executable, "-c", "pass"],
                cwd=tmp_path,
                env=dict(os.environ),
                output=tmp_path / "out",
                timeout=10,
                job_name=name,
            )
        limits = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        assert limits["BasicLimitInformation"]["LimitFlags"] == 0
    finally:
        job.Close()
