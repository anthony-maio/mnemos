"""Open the normal hook-review UI only for a hash-bound disposable profile."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4
import zipfile


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(manifest_path: Path, expected_sha256: str, *, check_only: bool) -> dict[str, Any]:
    if sha(manifest_path) != expected_sha256:
        raise ValueError("Review preparation changed; inspect it before continuing")
    manifest = json.loads(manifest_path.read_bytes())
    root = Path(manifest["root"]).resolve()
    home, project, pilot = (
        Path(manifest[key]).resolve() for key in ("codex_home", "project", "pilot_root")
    )
    repo, binary = Path(manifest["repository"]), Path(manifest["codex"])
    if manifest_path.resolve().parent != root or not all(
        path.is_relative_to(root) for path in (home, project)
    ):
        raise ValueError("Review paths escape the prepared directory")
    if root.is_relative_to(pilot) or home == Path.home() / ".codex":
        raise ValueError("Refusing a live pilot or ordinary Codex home")
    if Path(sys.executable).resolve() != Path(manifest["python"]).resolve():
        raise ValueError("Use the prepared installed Python")
    if (
        sha(binary) != manifest["codex_sha256"]
        or sha(binary.with_name("codex-code-mode-host.exe")) != manifest["code_mode_sha256"]
    ):
        raise ValueError("Native executable changed")
    for name, digest in manifest["source_helpers"].items():
        if sha(repo / name) != digest:
            raise ValueError("Review helper changed: " + name)
    for name, digest in manifest["fixed_files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError("Disposable runtime changed: " + name)
    if sha(home / "config.toml") != manifest["original_config_sha256"]:
        raise ValueError("Prepared configuration changed; review its diff before proceeding")
    if (pilot / "memory.sqlite").exists() or (pilot / "runtime").exists():
        raise ValueError("The actual pilot must remain inactive")
    watched = {Path(name): digest for name, digest in manifest["watched_paths"].items()}
    if any((sha(path) if path.is_file() else None) != digest for path, digest in watched.items()):
        raise ValueError("A live reference changed since preparation")

    import mnemos

    package = Path(mnemos.__file__).resolve().parent
    if not package.is_relative_to(pilot / "venv"):
        raise ValueError("Package escaped the prepared installation")
    wheel = pilot / "artifacts/mnemos_memory-0.7.0b1-py3-none-any.whl"
    if sha(wheel) != manifest["wheel_sha256"]:
        raise ValueError("Installed artifact binding changed")
    with zipfile.ZipFile(wheel) as stream:
        for name in stream.namelist():
            if name.startswith("mnemos/") and not name.endswith("/"):
                if (package.parent / name).read_bytes() != stream.read(name):
                    raise ValueError("Installed package changed: " + name)
    sys.path.insert(0, str(repo))
    from benchmarks.native_hosts.codex_hook_trust import read_inventory, validate_inventory
    from benchmarks.native_hosts.native_processes import (
        _auth_copy,
        clean_environment,
        terminate_owned_process,
    )
    from benchmarks.native_hosts.pilot_outage import remaining_service_pids

    attempt = root / ("review-check-" if check_only else "user-review-") / uuid4().hex
    attempt.mkdir(parents=True)
    record: dict[str, Any] = {
        "status": "preflight",
        "check_only": check_only,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    result_path = attempt / "result.json"

    def save() -> None:
        result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    raw = read_inventory(binary, home=home, project=project, output=attempt / "before")
    record["before"] = validate_inventory(
        raw,
        project=project,
        config_path=home / "config.toml",
        expected_hooks=manifest["expected_hooks"],
        expected_hashes=manifest["inventory"]["current_hashes"],
    )
    if check_only:
        record.update(
            status="ready_for_user_review", credentials_copied=False, interactive_host_started=False
        )
        save()
        return record

    checkpoint = runpy.run_path(
        str(repo / "scripts/check-native-explicit.py"), run_name="review_helpers"
    )
    auth = Path.home() / ".codex/auth.json"
    record["authentication_preflight"] = checkpoint["codex_auth_preflight"](auth)
    auth_hash = sha(auth)
    copied = None
    process = None
    env = clean_environment({"CODEX_HOME": str(home)})
    disabled = [
        "shell_tool",
        "shell_snapshot",
        "unified_exec",
        "apps",
        "multi_agent",
        "multi_agent_v2",
        "memories",
        "goals",
        "plugins",
        "remote_plugin",
        "browser_use",
        "browser_use_external",
        "computer_use",
        "image_generation",
        "artifact",
        "workspace_dependencies",
        "skill_search",
    ]
    command = [
        str(binary),
        "--no-daemon",
        "--no-alt-screen",
        "--ask-for-approval",
        "on-request",
        "--sandbox",
        "read-only",
        "--enable",
        "hooks",
        "--enable",
        "code_mode_host",
        *(value for feature in disabled for value in ("--disable", feature)),
        "-c",
        "mcp_servers.mnemos.enabled=false",
        "-c",
        'web_search="disabled"',
        "-c",
        'cli_auth_credentials_store="file"',
        "--model",
        "gpt-6-astra",
        "--cd",
        str(project),
    ]
    record.update(status="awaiting_native_user_review", command=command)
    save()
    print(
        "Open /hooks. Review the six Mnemos hooks, then trust them if their commands match REVIEW.md.",
        flush=True,
    )
    print(
        "After reviewing, exit Codex without submitting a model prompt. This profile uses disposable data.",
        flush=True,
    )
    try:
        copied = _auth_copy(auth, home, root)
        process = subprocess.Popen(command, cwd=project, env=env)
        record["pid"] = process.pid
        save()
        try:
            process.wait(timeout=1200)
        except KeyboardInterrupt:
            record["terminal_interrupted"] = True
        except subprocess.TimeoutExpired:
            record["review_timed_out"] = True
    finally:
        if process is not None:
            terminate_owned_process(process)
            record["returncode"] = process.returncode
        if copied is not None:
            if copied.resolve().parent != home or copied.name != "auth.json":
                raise ValueError("Authentication cleanup path escaped the isolated home")
            copied.unlink(missing_ok=True)
        record["copied_auth_removed"] = not (home / "auth.json").exists()
        # A trusted SessionEnd hook may start this disposable service on exit.
        descriptor = root / "runtime/runtime.json"
        for _ in range(10):
            if descriptor.exists():
                break
            time.sleep(0.25)
        if descriptor.exists():
            owner = json.loads(descriptor.read_bytes())
            record["disposable_service_pid"] = owner["pid"]
            stopped = subprocess.run(
                [sys.executable, "-I", str(root / "launch.py"), "cli", "codex", "service-stop"],
                cwd=project,
                env=clean_environment({}),
                capture_output=True,
                text=True,
                timeout=60,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            record["service_stop_returncode"] = stopped.returncode
            deadline = time.monotonic() + 20
            while descriptor.exists() and time.monotonic() < deadline:
                time.sleep(0.1)
        record["descriptor_removed"] = not descriptor.exists()
        record["remaining_disposable_service_pids"] = remaining_service_pids(descriptor)
        record["normal_credentials_unchanged"] = sha(auth) == auth_hash
        record["watched_paths_unchanged"] = all(
            (sha(p) if p.is_file() else None) == digest for p, digest in watched.items()
        )
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
    if (
        not all(
            record[k]
            for k in [
                "copied_auth_removed",
                "descriptor_removed",
                "normal_credentials_unchanged",
                "watched_paths_unchanged",
            ]
        )
        or record["remaining_disposable_service_pids"]
    ):
        raise ValueError("Review cleanup or original-file preservation failed")
    raw = read_inventory(binary, home=home, project=project, output=attempt / "after")
    record["after"] = validate_inventory(
        raw,
        project=project,
        config_path=home / "config.toml",
        expected_hooks=manifest["expected_hooks"],
        expected_hashes=manifest["inventory"]["current_hashes"],
    )
    record["observed_turn_contexts"] = sum(
        1
        for path in (home / "sessions").rglob("*.jsonl")
        for line in path.read_text("utf-8").splitlines()
        if line.strip() and json.loads(line).get("type") == "turn_context"
    )
    record["status"] = (
        "reviewed" if record["after"]["status"] == "trusted" else "review_still_required"
    )
    if record["observed_turn_contexts"]:
        record["status"] = "unexpected_model_turn_requires_inspection"
    save()
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = run(args.manifest, args.expected_sha256, check_only=args.check_only)
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in [
                    "status",
                    "check_only",
                    "copied_auth_removed",
                    "descriptor_removed",
                    "observed_turn_contexts",
                ]
            }
        )
    )
