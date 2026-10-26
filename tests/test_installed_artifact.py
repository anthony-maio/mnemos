"""Isolation and release checks for the installed-package example."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(path: str) -> Any:
    spec = importlib.util.spec_from_file_location("installed_example", ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_replaces_live_memory_configuration(monkeypatch: Any, tmp_path: Path) -> None:
    example = load_script("examples/restart_recall.py")
    inherited = {
        "MNEMOS_SQLITE_PATH": "live.sqlite",
        "MNEMOS_RUNTIME_DESCRIPTOR": "live-service.json",
        "MNEMOS_CONFIG_PATH": "live.toml",
        "MNEMOS_SCOPE_REGISTRY": "live-scopes.json",
        "MNEMOS_PROJECT_ROOT": "live-project",
        "MNEMOS_TYPESAFE_API_KEY": "private-placeholder",
        "MNEMOS_AUTO_START": "1",
        "PYTHONPATH": "source-checkout",
        "GIT_DIR": "live-git-directory",
    }
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)
    env = example.isolated_env(tmp_path)
    for key in (
        "MNEMOS_SQLITE_PATH",
        "MNEMOS_RUNTIME_DESCRIPTOR",
        "MNEMOS_CONFIG_PATH",
        "MNEMOS_SCOPE_REGISTRY",
        "MNEMOS_RUNTIME_DIR",
    ):
        assert Path(env[key]).is_relative_to(tmp_path)
    assert "MNEMOS_PROJECT_ROOT" not in env
    assert "MNEMOS_TYPESAFE_API_KEY" not in env
    assert "PYTHONPATH" not in env
    assert "GIT_DIR" not in env
    assert env["MNEMOS_AUTO_START"] == "0"


def test_recall_example_rejects_wrong_identity_scope_and_lost_evidence() -> None:
    example = load_script("examples/restart_recall.py")
    fact = {
        "id": "original-id",
        "content": example.FACT,
        "scope": "project",
        "scope_id": "notebook",
        "sources": [{"id": "evidence"}],
        "support_available": True,
    }
    example.assert_recalled({"facts": [fact]}, "original-id", "notebook")
    for changes in (
        {"id": "replacement-id"},
        {"scope_id": "unrelated-project"},
        {"content": "A different fact"},
        {"sources": []},
        {"support_available": False},
    ):
        with pytest.raises(AssertionError):
            example.assert_recalled({"facts": [{**fact, **changes}]}, "original-id", "notebook")


def test_smoke_reads_wheel_metadata_and_rejects_another_distribution(tmp_path: Path) -> None:
    smoke = load_script("scripts/smoke-installed.py")
    wheel = tmp_path / "example.whl"
    for name in ("mnemos-memory", "another-package"):
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(
                "example.dist-info/METADATA",
                f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.7.0b1\n",
            )
        if name == "mnemos-memory":
            assert smoke.wheel_version(wheel) == "0.7.0b1"
        else:
            with pytest.raises(ValueError, match="mnemos-memory"):
                smoke.wheel_version(wheel)


def test_pypi_upload_keeps_environment_and_validation_gates() -> None:
    workflow = (ROOT / ".github/workflows/pypi.yml").read_text(encoding="utf-8")
    upload = workflow.index("pypa/gh-action-pypi-publish")
    assert "environment: mnemos-release" in workflow
    for gate in (
        "black --check .",
        "mypy .",
        "pytest -q",
        "--enforce-production-gate",
        "python -m twine check",
        "scripts/smoke-installed.py",
        '--release-tag "$RELEASE_TAG"',
    ):
        assert workflow.index(gate) < upload
    assert "continue-on-error" not in workflow


def test_ci_release_readiness_waits_for_installed_artifact() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    readiness = workflow.split("  release_readiness:", 1)[1].split("  installed_artifact:", 1)[0]
    assert "needs: [benchmark_gate, tier1_mcp_smoke, installed_artifact]" in readiness
    assert "scripts/smoke-installed.py --wheel dist/*.whl" in workflow
