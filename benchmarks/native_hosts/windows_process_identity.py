"""Identify a disposable service process behind a Windows venv launcher."""

from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def process_snapshot(pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError("A positive integer PID is required")
    command = (
        'Get-CimInstance Win32_Process -Filter "ProcessId=' + str(pid) + '" | '
        "Select-Object ProcessId,ParentProcessId,Name,CreationDate,ExecutablePath,CommandLine | "
        "ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def command_arguments(command):
    count = ctypes.c_int()
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    parse = shell.CommandLineToArgvW
    parse.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    parse.restype = ctypes.POINTER(ctypes.c_wchar_p)
    free = kernel.LocalFree
    free.argtypes = [ctypes.c_void_p]
    free.restype = ctypes.c_void_p
    values = parse(command, ctypes.byref(count))
    if not values:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return [values[index] for index in range(count.value)]
    finally:
        free(ctypes.cast(values, ctypes.c_void_p))


def normalized_path(value):
    return os.path.normcase(str(Path(value).resolve()))


def validate_snapshot(snapshot, *, launcher_pid, driver_pid, entry, executable):
    if not snapshot:
        raise RuntimeError("Service interpreter is absent")
    owner_pid = snapshot["ProcessId"]
    expected_parent = driver_pid if owner_pid == launcher_pid else launcher_pid
    if snapshot["ParentProcessId"] != expected_parent:
        raise RuntimeError("Service interpreter has an unexpected parent")
    if normalized_path(snapshot["ExecutablePath"]) != normalized_path(executable):
        raise RuntimeError("Service interpreter executable differs")
    arguments = command_arguments(snapshot["CommandLine"])
    if (
        len(arguments) != 3
        or arguments[1] != "-I"
        or normalized_path(arguments[0]) != normalized_path(executable)
        or normalized_path(arguments[2]) != normalized_path(entry)
    ):
        raise RuntimeError("Service interpreter command differs")
    if not snapshot.get("CreationDate"):
        raise RuntimeError("Service interpreter has no creation timestamp")
    return {
        "owner_pid": owner_pid,
        "launcher_pid": launcher_pid,
        "relation": "direct" if owner_pid == launcher_pid else "direct_child",
        "entry": str(Path(entry).resolve()),
        "executable": str(Path(executable).resolve()),
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "process": snapshot,
    }


def verify_owner(launcher, owner_pid, entry):
    if launcher.poll() is not None:
        raise RuntimeError("Service launcher already exited")
    return validate_snapshot(
        process_snapshot(owner_pid),
        launcher_pid=launcher.pid,
        driver_pid=os.getpid(),
        entry=entry,
        executable=sys._base_executable,
    )


def same_owner(proof):
    current = process_snapshot(proof["owner_pid"])
    if current is None:
        return False
    previous = proof["process"]
    return all(
        current.get(field) == previous.get(field)
        for field in (
            "ProcessId",
            "ParentProcessId",
            "CreationDate",
            "ExecutablePath",
            "CommandLine",
        )
    )
