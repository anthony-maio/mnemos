from __future__ import annotations

from pathlib import Path


def test_ci_runs_shipped_repeat_benchmark_gate() -> None:
    workflow = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "benchmark_gate:" in workflow
    assert "--stores memory,sqlite" in workflow
    assert "--dataset-pack claim-driving" in workflow
    assert "--repetitions 2" in workflow
    assert "--enforce-production-gate" in workflow
    assert "qdrant" not in workflow
