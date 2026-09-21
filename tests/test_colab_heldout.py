"""The Colab held-out runner verifies its bundle, resumes from Drive, and pairs results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.local_memory import colab_heldout as colab

DIGEST = {"qwen": "a" * 64, "nomic": "b" * 64}


def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    bundle, source = tmp_path / "bundle", tmp_path / "source"
    (source / "mnemos").mkdir(parents=True)
    # Bytes, not text: digests must not depend on the platform's line endings.
    (source / "mnemos" / "__init__.py").write_bytes(b"VALUE = 1\n")
    bundle.mkdir()
    (bundle / "corpus.jsonl").write_bytes(b'{"record_type": "manifest"}\n')
    preparation = {
        "source_commit": "c" * 40,
        "source_files": {
            "mnemos/__init__.py": hashlib.sha256(b"VALUE = 1\n").hexdigest(),
        },
        "corpus_sha256": hashlib.sha256(b'{"record_type": "manifest"}\n').hexdigest(),
        "heldout_sessions": 200,
        "repetitions": 3,
        "model": "qwen",
        "embedding_model": "nomic",
        "models": {name: {"digest": digest} for name, digest in DIGEST.items()},
        "options": {
            "think": False,
            "temperature": 0.0,
            "max_tokens": 8192,
            "timeout_seconds": 180.0,
            "extraction_mode": "source_bound",
        },
        "run_namespace": "colab-test",
    }
    (bundle / "preparation.json").write_text(json.dumps(preparation), encoding="utf-8")
    return bundle, source


def test_a_changed_source_or_corpus_is_refused(tmp_path: Path) -> None:
    bundle, source = _bundle(tmp_path)
    colab.verify_bundle(bundle, source)

    (source / "mnemos" / "__init__.py").write_bytes(b"VALUE = 2\n")
    with pytest.raises(RuntimeError, match="mnemos/__init__.py"):
        colab.verify_bundle(bundle, source)
    (source / "mnemos" / "__init__.py").write_bytes(b"VALUE = 1\n")
    (bundle / "corpus.jsonl").write_bytes(b"{}\n")
    with pytest.raises(RuntimeError, match="corpus"):
        colab.verify_bundle(bundle, source)


def _patch_run(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]], live: dict) -> None:
    monkeypatch.setattr(
        colab,
        "_live_model_metadata",
        lambda endpoint, names: {name: {"digest": live[name]} for name in names},
    )
    monkeypatch.setattr(
        colab,
        "server_binding",
        lambda endpoint: {
            "gpu": "NVIDIA H100 80GB HBM3",
            "ollama_version": "0.34.2",
            "ollama_context_length": "16384",
            "ollama_num_parallel": "8",
        },
    )

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        work = kwargs["work_dir"]
        calls.append({**kwargs, "restored": (work / "full.checkpoint.json").is_file()})
        (work / "full.checkpoint.json").write_text(
            json.dumps({"completed_observation_ids": ["one", "two"]}), encoding="utf-8"
        )
        report = kwargs["output_dir"] / "full.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        return {"full": {"status": "completed"}}

    monkeypatch.setattr(colab, "run_heldout", fake_run)


def test_a_relaunch_resumes_from_the_mirrored_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, source = _bundle(tmp_path)
    run, mirror = tmp_path / "vm", tmp_path / "drive"
    (mirror / "work").mkdir(parents=True)
    (mirror / "work" / "full.checkpoint.json").write_text("{}", encoding="utf-8")
    calls: list[dict[str, Any]] = []
    _patch_run(monkeypatch, calls, DIGEST)

    status = colab.run_profile(
        bundle=bundle, source=source, run=run, mirror=mirror, profile="full", endpoint="x"
    )

    assert calls[0]["restored"] is True
    assert calls[0]["profile_names"] == {"full"}
    assert calls[0]["extraction_mode"] == "source_bound"
    binding = calls[0]["experiment_binding"]
    assert binding["gpu"] == "NVIDIA H100 80GB HBM3"
    assert (
        binding["preparation_sha256"]
        == hashlib.sha256((bundle / "preparation.json").read_bytes()).hexdigest()
    )
    assert status["status"] == "completed" and status["completed_observations"] == 2
    mirrored = json.loads((mirror / "status" / "full.json").read_text(encoding="utf-8"))
    assert mirrored["status"] == "completed" and mirrored["expected_observations"] == 600
    assert json.loads((mirror / "reports" / "full.json").read_text())["status"] == "completed"


def test_different_weights_are_refused_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, source = _bundle(tmp_path)
    calls: list[dict[str, Any]] = []
    _patch_run(monkeypatch, calls, {**DIGEST, "qwen": "d" * 64})

    with pytest.raises(RuntimeError, match="qwen digest"):
        colab.run_profile(
            bundle=bundle,
            source=source,
            run=tmp_path / "vm",
            mirror=tmp_path / "drive",
            profile="full",
            endpoint="x",
        )
    assert calls == []


def _report(hits: list[bool]) -> dict[str, Any]:
    return {
        "status": "completed",
        "raw_outputs": [
            {
                "session_id": f"s{index}",
                "repetition": 0,
                "execution_status": "complete",
                "answerable_hits": [hit],
                "no_answer_correct": [True],
                "critical_retention": [hit],
                "critical_retrieval": [hit],
            }
            for index, hit in enumerate(hits)
        ],
    }


def test_aggregate_pairs_finished_profiles_with_the_baseline(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "baseline.json").write_text(json.dumps(_report([False] * 6)), encoding="utf-8")
    (reports / "full.json").write_text(json.dumps(_report([True] * 6)), encoding="utf-8")

    summary = colab.aggregate(mirror=tmp_path, repetitions=3)

    assert summary["statuses"] == {"baseline": "completed", "full": "completed"}
    assert "ablate-surprisal_gating" in summary["missing_profiles"]
    hits = summary["paired_comparisons"]["full"]["metrics"]["answerable_hits"]
    assert hits["estimate"] == pytest.approx(1.0)
    stored = json.loads((reports / "baseline.json").read_text(encoding="utf-8"))
    assert stored["paired_comparisons"] == summary["paired_comparisons"]
    assert json.loads((tmp_path / "summary.json").read_text())["statuses"] == summary["statuses"]


def test_a_second_runner_for_the_same_profile_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, source = _bundle(tmp_path)
    calls: list[dict[str, Any]] = []
    _patch_run(monkeypatch, calls, DIGEST)
    run = tmp_path / "vm"

    # A runner that outlived a kernel restart still holds its profile.
    with colab._profile_lock(run, "full"):
        with pytest.raises(RuntimeError, match="already holds profile 'full'"):
            colab.run_profile(
                bundle=bundle,
                source=source,
                run=run,
                mirror=tmp_path / "drive",
                profile="full",
                endpoint="x",
            )
    assert calls == []
    assert not (tmp_path / "drive" / "status" / "full.json").exists()
