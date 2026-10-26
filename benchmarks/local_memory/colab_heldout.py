"""Run one held-out profile on a cloud GPU VM and mirror its progress to Drive.

The notebook in benchmarks/local_memory/colab starts one of these per profile,
all against one local Ollama server. Work stays on the VM's own disk: the
checkpoint is rewritten after every observation and each attempt database
grows to gigabytes, which a Drive mount cannot absorb. A mirror thread copies
the small durable state to Drive every few minutes, and a relaunch on a new VM
restores it before resuming. Profiles write disjoint files, so they can run in
parallel; ``aggregate`` adds the paired comparisons once all of them finish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.request import urlopen

from .artifacts import ArtifactRecorder
from .harness import DEFAULT_PROFILES, ExperimentConfig, compare_observation_metrics
from .heldout_run import _live_model_metadata, run_heldout

MIRROR_INTERVAL_SECONDS = 300.0
PROFILE_NAMES = tuple(profile.name for profile in DEFAULT_PROFILES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _copy(source: Path, target: Path) -> None:
    """Copy through a temporary name so Drive never holds half a file."""
    if not source.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.partial")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def verify_bundle(bundle: Path, source: Path) -> dict[str, Any]:
    """Refuse to run anything but the prepared source and corpus bytes."""
    preparation: dict[str, Any] = json.loads(
        (bundle / "preparation.json").read_text(encoding="utf-8")
    )
    if _sha256(bundle / "corpus.jsonl") != preparation["corpus_sha256"]:
        raise RuntimeError("corpus.jsonl does not match preparation.json")
    for name, digest in preparation["source_files"].items():
        path = source / name
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"source file {name} does not match preparation.json")
    return preparation


def server_binding(endpoint: str) -> dict[str, str]:
    """Settings outside the model files that can still change its outputs."""
    with urlopen(endpoint.rstrip("/") + "/api/version", timeout=30.0) as response:  # noqa: S310
        ollama_version = str(json.loads(response.read().decode("utf-8")).get("version"))
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "gpu": gpu.stdout.strip().splitlines()[0] if gpu.returncode == 0 and gpu.stdout else "",
        "ollama_version": ollama_version,
        "ollama_context_length": os.environ.get("OLLAMA_CONTEXT_LENGTH", ""),
        "ollama_num_parallel": os.environ.get("OLLAMA_NUM_PARALLEL", ""),
    }


def _profile_files(run: Path, profile: str) -> list[Path]:
    return [
        run / "work" / f"{profile}.checkpoint.json",
        run / "work" / f"{profile}.manifest.json",
        run / "status" / f"{profile}.json",
        run / "reports" / f"{profile}.json",
        run / "reports" / f"{profile}.raw-outputs.jsonl",
        run / "logs" / f"{profile}.log",
    ]


def mirror_profile(run: Path, mirror: Path, profile: str) -> None:
    for path in _profile_files(run, profile):
        _copy(path, mirror / path.relative_to(run))


def restore_profile(run: Path, mirror: Path, profile: str) -> bool:
    """Bring a mirrored checkpoint back onto a fresh VM before resuming."""
    saved = mirror / "work" / f"{profile}.checkpoint.json"
    local = run / "work" / f"{profile}.checkpoint.json"
    if not saved.is_file() or local.is_file():
        return False
    _copy(saved, local)
    return True


def _completed_observations(checkpoint: Path) -> int | None:
    try:
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    completed = payload.get("completed_observation_ids") if isinstance(payload, dict) else None
    return len(completed) if isinstance(completed, list) else None


@contextmanager
def _profile_lock(run: Path, profile: str) -> Iterator[None]:
    """Refuse a second runner for one profile, such as after a kernel restart."""
    path = run / "work" / f"{profile}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"another runner already holds profile {profile!r}") from exc
        yield


def run_profile(
    *, bundle: Path, source: Path, run: Path, mirror: Path, profile: str, endpoint: str
) -> dict[str, Any]:
    if profile not in PROFILE_NAMES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILE_NAMES}")
    with _profile_lock(run, profile):
        return _run_locked(
            bundle=bundle, source=source, run=run, mirror=mirror, profile=profile, endpoint=endpoint
        )


def _run_locked(
    *, bundle: Path, source: Path, run: Path, mirror: Path, profile: str, endpoint: str
) -> dict[str, Any]:
    preparation = verify_bundle(bundle, source)
    live = _live_model_metadata(endpoint, (preparation["model"], preparation["embedding_model"]))
    for name, record in preparation["models"].items():
        if live[name]["digest"] != record["digest"]:
            raise RuntimeError(f"live {name} digest differs from the prepared model")
    binding = {
        "preparation_sha256": _sha256(bundle / "preparation.json"),
        "source_commit": preparation["source_commit"],
        **server_binding(endpoint),
    }
    restored = restore_profile(run, mirror, profile)
    options = preparation["options"]
    status_path = run / "status" / f"{profile}.json"
    status: dict[str, Any] = {
        "profile": profile,
        "status": "running",
        "pid": os.getpid(),
        "started_at": _now(),
        "restored_checkpoint": restored,
        "binding": binding,
        "expected_observations": preparation["heldout_sessions"] * preparation["repetitions"],
    }

    def publish() -> None:
        status["updated_at"] = _now()
        status["completed_observations"] = _completed_observations(
            run / "work" / f"{profile}.checkpoint.json"
        )
        _write_json(status_path, status)
        mirror_profile(run, mirror, profile)

    stop = threading.Event()

    def mirror_loop() -> None:
        while not stop.wait(MIRROR_INTERVAL_SECONDS):
            try:
                publish()
            except OSError:
                # A Drive hiccup must not end the run; the next pass retries.
                pass

    publish()
    thread = threading.Thread(target=mirror_loop, name=f"mirror-{profile}", daemon=True)
    thread.start()
    try:
        reports = run_heldout(
            dataset_path=bundle / "corpus.jsonl",
            output_dir=run / "reports",
            work_dir=run / "work",
            endpoint=endpoint,
            model=preparation["model"],
            embedding_model=preparation["embedding_model"],
            repetitions=preparation["repetitions"],
            profile_names={profile},
            root=source,
            think=options["think"],
            temperature=options["temperature"],
            max_tokens=options["max_tokens"],
            timeout_seconds=options["timeout_seconds"],
            extraction_mode=options["extraction_mode"],
            run_namespace=preparation["run_namespace"],
            experiment_binding=binding,
            judge_provider=options.get("judge_provider", "none"),
            typesafe_api_key=os.environ.get("MNEMOS_TYPESAFE_API_KEY"),
            typesafe_model=options.get("typesafe_model", "jev-latest"),
            surprisal_mode=options.get("surprisal_mode", "prediction"),
        )
        status["status"] = reports[profile].get("status", "incomplete")
    except BaseException as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:2000])
        raise
    finally:
        stop.set()
        thread.join(timeout=60)
        status["finished_at"] = _now()
        publish()
    return status


def _profile_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize complete extraction observations and all correction attempts."""
    observations = report.get("raw_outputs", [])
    complete = [row for row in observations if row.get("execution_status") == "complete"]
    precision = [
        value
        for row in complete
        for value in row.get("supported_precision", [])
        if isinstance(value, bool)
    ]
    correction_attempts = correction_commits = 0
    for row in observations:
        lifecycle = row.get("raw_output", {}).get("lifecycle_summary", {})
        correction = lifecycle.get("correct", {})
        correction_attempts += int(correction.get("attempted", 0))
        correction_commits += int(correction.get("committed", 0))
    return {
        "supported_precision": {
            "estimate": sum(precision) / len(precision) if precision else None,
            "sample_count": len(precision),
            "complete_observations": len(complete),
            "excluded_incomplete_observations": len(observations) - len(complete),
        },
        "correction_commit_rate": {
            "estimate": correction_commits / correction_attempts if correction_attempts else None,
            "committed": correction_commits,
            "attempted": correction_attempts,
        },
    }


def aggregate(*, mirror: Path, repetitions: int) -> dict[str, Any]:
    """Summarize present profiles and pair them when a baseline is available."""
    reports = mirror / "reports"
    baseline_path = reports / "baseline.json"
    baseline = (
        json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else None
    )
    config = ExperimentConfig(repetitions=repetitions)
    comparisons: dict[str, Any] = {}
    statuses: dict[str, Any] = {}
    profile_metrics: dict[str, Any] = {}
    for name in PROFILE_NAMES:
        path = reports / f"{name}.json"
        if not path.is_file():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        statuses[name] = report.get("status")
        profile_metrics[name] = _profile_metrics(report)
        if baseline is not None and name != "baseline":
            comparisons[name] = compare_observation_metrics(baseline, report, config=config)
    if baseline is not None:
        baseline["paired_comparisons"] = comparisons
        ArtifactRecorder(baseline_path).write(baseline)
    summary = {
        "written_at": _now(),
        "statuses": statuses,
        "missing_profiles": [name for name in PROFILE_NAMES if name not in statuses],
        "profile_metrics": profile_metrics,
        "paired_comparisons": comparisons,
    }
    _write_json(mirror / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_command = commands.add_parser("run", help="run or resume one profile")
    run_command.add_argument("--bundle", type=Path, required=True)
    run_command.add_argument("--source", type=Path, required=True)
    run_command.add_argument("--run", type=Path, required=True)
    run_command.add_argument("--mirror", type=Path, required=True)
    run_command.add_argument("--profile", choices=PROFILE_NAMES, required=True)
    run_command.add_argument("--endpoint", default="http://127.0.0.1:11434")
    aggregate_command = commands.add_parser("aggregate", help="pair profiles with baseline")
    aggregate_command.add_argument("--mirror", type=Path, required=True)
    aggregate_command.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_profile(
            bundle=args.bundle,
            source=args.source,
            run=args.run,
            mirror=args.mirror,
            profile=args.profile,
            endpoint=args.endpoint,
        )
    else:
        result = aggregate(mirror=args.mirror, repetitions=args.repetitions)
    print(json.dumps(result, sort_keys=True, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
