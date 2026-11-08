"""Bound disposable Windows host descendants without modifying plugin launchers."""

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

BROKER = """import json,subprocess,sys
payload=json.loads(sys.stdin.readline())
completed=subprocess.run(payload['command'],cwd=payload['cwd'],env=payload['env'],stdin=subprocess.DEVNULL)
raise SystemExit(completed.returncode)
"""


def run_owned_job(command, *, cwd, env, output, timeout, job_name=None):
    """The broker waits for assignment to a new job, unnamed by default.

    No breakaway flags are enabled. Cleanup acts on this job and its owned
    Popen handle only, never a process name or a rediscovered PID.
    """
    if os.name != "nt":
        raise RuntimeError("Windows job containment required")
    if not 0 < timeout <= 300:
        raise ValueError("Expected a bounded native phase timeout")
    import win32api
    import win32job

    output = Path(output)
    output.mkdir(exist_ok=False)
    result = dict(
        status="failed", job_assigned=False, input_delivery_started=False, job_name=job_name
    )
    job = process = failure = writer = None
    cleanup_errors = []
    started = time.perf_counter()
    try:
        job = win32job.CreateJobObject(None, job_name or "")
        if job_name and win32api.GetLastError() == 183:
            # Never reconfigure or terminate a pre-existing named job.
            job.Close()
            job = None
            raise RuntimeError("Named supervisor job already exists")
        limits = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        limits["BasicLimitInformation"]["LimitFlags"] = win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, limits)
        with (
            (output / "stdout.log").open("xb") as stdout,
            (output / "stderr.log").open("xb") as stderr,
        ):
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", BROKER],
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            # Use the exact process handle, not a potentially reused PID lookup.
            win32job.AssignProcessToJobObject(job, int(process._handle))
            result.update(job_assigned=True, broker_pid=process.pid)
            payload = dict(command=command, cwd=str(cwd), env=env)
            encoded = (json.dumps(payload) + "\n").encode("utf-8")
            write_errors = []

            def deliver():
                try:
                    process.stdin.write(encoded)
                    process.stdin.close()
                except Exception as exc:
                    write_errors.append(exc)

            deadline = time.monotonic() + timeout
            result["input_delivery_started"] = True
            writer = threading.Thread(target=deliver, daemon=True)
            writer.start()
            writer.join(timeout=timeout)
            if writer.is_alive():
                raise subprocess.TimeoutExpired(command, timeout)
            if write_errors:
                raise write_errors[0]
            result["exit_code"] = process.wait(timeout=max(0, deadline - time.monotonic()))
            if result["exit_code"] != 0:
                raise RuntimeError("Native command exited unsuccessfully")
            result["status"] = "passed"
    except BaseException as exc:
        failure = exc
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if job is not None:
            try:
                info = win32job.QueryInformationJobObject(
                    job, win32job.JobObjectBasicAccountingInformation
                )
                result["active_before_cleanup"] = info["ActiveProcesses"]
                result["total_job_processes"] = info["TotalProcesses"]
                win32job.TerminateJobObject(job, 1)
                deadline = time.monotonic() + 10
                while True:
                    active = win32job.QueryInformationJobObject(
                        job, win32job.JobObjectBasicAccountingInformation
                    )["ActiveProcesses"]
                    if not active or time.monotonic() >= deadline:
                        break
                    time.sleep(0.02)
                result["active_after_cleanup"] = active
                if active:
                    cleanup_errors.append("Job still has active processes")
            except Exception as exc:
                cleanup_errors.append("Job cleanup: " + type(exc).__name__)
            finally:
                try:
                    job.Close()
                except Exception as exc:
                    cleanup_errors.append("Job close: " + type(exc).__name__)
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
                if writer is not None:
                    writer.join(timeout=10)
                    if writer.is_alive():
                        raise RuntimeError("Input writer did not exit after broker termination")
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
            except Exception as exc:
                cleanup_errors.append("Broker cleanup: " + type(exc).__name__)
        if cleanup_errors:
            result.update(status="failed", cleanup_errors=cleanup_errors)
        result["elapsed_seconds"] = time.perf_counter() - started
        (output / "process.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if failure is not None:
        raise failure
    if cleanup_errors:
        raise RuntimeError("Owned process cleanup failed")
    return result
