"""Offline runtime boundaries using the installed report schema."""

import asyncio
import json
import zipfile
from types import SimpleNamespace

import pytest
from benchmarks.native_hosts import three_host_runtime as m


@pytest.fixture
def fx(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "docs/plans").mkdir(parents=True)
    source = tmp_path / "source"
    files = {
        n: b"verified plugin"
        for n in [
            ".claude-plugin/plugin.json",
            ".claude-plugin/run_mnemos_hook.py",
            ".claude-plugin/run_mnemos_mcp.py",
            "hooks/hooks.json",
        ]
    }
    for n, raw in files.items():
        p = source / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    input_bytes = {relative: relative.encode() for relative in m.HERMES_INPUTS}
    for relative, raw in input_bytes.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    env = tmp_path / "installed"
    package = env / "Lib/site-packages/mnemos/__init__.py"
    package.parent.mkdir(parents=True)
    package.write_bytes(b"verified package")
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("mnemos/__init__.py", b"verified package")
    digest = m.sha(wheel)
    monkeypatch.setattr(m, "WHEEL_SHA256", digest)
    (repo / "docs/plans/2026-09-19-three-host-continuity-protocol-v1.md").write_text(
        f"{m.SOURCE_COMMIT} {digest}"
    )

    def report(name, value):
        p = tmp_path / (name + ".json")
        p.write_text(json.dumps(value))
        return p

    checks = report(
        "checks",
        {
            "status": "passed",
            "source_commit": m.SOURCE_COMMIT,
            "source_checkout": str(source),
            "artifacts": [{"file": str(wheel), "sha256": digest}],
        },
    )
    installed = report(
        "installed",
        {
            "status": "passed",
            "environment": str(env),
            "bindings_after": {
                "commit": m.SOURCE_COMMIT,
                "wheel_sha256": digest,
                "checks_sha256": m.sha(checks),
            },
        },
    )
    names = [
        "teach",
        "recall_v1",
        "correct",
        "recall_v2",
        "forget",
        "recall_forgotten",
        "source-replay",
    ]
    gates = {
        name: True
        for name in [
            "correlated",
            "phase_tools_allowed",
            "mutations_inspected",
            "no_foreign_tool_result",
            "real_pinned_model_request",
            "pinned_profile_without_conflicting_override",
            "write_followup_guidance_present",
            "no_provider_error",
            "controls_unchanged",
            "native_capture_committed",
            "native_capture_retained_or_suppressed",
            "teach",
            "single_asserted_target",
            "recall_v1",
            "correct",
            "same_id_version_two",
            "recall_v2",
            "forget",
            "target_absent",
            "target_suppressed",
            "recall_forgotten",
            "no_forgotten_label_in_request",
            "original_write_suppressed",
            "native_replay_flushed",
        ]
    }
    hermes = report(
        "hermes",
        {
            "status": "passed",
            "source_commit": m.SOURCE_COMMIT,
            "wheel_sha256": digest,
            "runner_commit": "fixture-commit",
            "input_hashes": {relative: m.sha(repo / relative) for relative in m.HERMES_INPUTS},
            "phases": [{"phase": n, "exit_code": 0, "gates": gates} for n in names],
        },
    )
    queue = report("queue", {"pid": 123, "status": "finished"})
    monkeypatch.setattr(
        m, "require_prior_queue_terminal", lambda path, pids: {"excluded_pids": pids}
    )
    monkeypatch.setattr(m.sys, "prefix", str(env))
    import mnemos

    monkeypatch.setattr(mnemos, "__file__", str(package))

    def git(command, **kwargs):
        if "ls-tree" in command:
            return "\n".join(files)
        if "show" in command:
            relative = command[-1].split(":", 1)[1]
            return files[relative] if relative in files else input_bytes[relative]
        raise AssertionError("unexpected offline subprocess")

    monkeypatch.setattr(m.subprocess, "check_output", git)
    protected = []
    monkeypatch.setattr("mnemos.service._protect_private_path", lambda p: protected.append(p))
    rt = m.ThreeHostRuntime(tmp_path / "runtime", repo, checks, installed, queue, [123], hermes)
    return SimpleNamespace(rt=rt, package=package, source=source, protected=protected)


def test_hermes_prerequisite_requires_recorded_write_guidance(fx):
    path = fx.rt.hermes_acceptance
    report = json.loads(path.read_text())
    del report["phases"][0]["gates"]["write_followup_guidance_present"]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="required successful phase gates"):
        fx.rt.verify()


def test_real_report_shape_and_verified_copy(fx):
    bindings = fx.rt.verify()
    fx.rt.prepare()
    assert fx.protected[0] == fx.rt.root
    for name, digest in bindings["plugin_files"].items():
        assert m.sha(fx.rt.plugin_snapshot / name) == digest
    assert (
        fx.rt.contexts["project"]["project_id"] != fx.rt.contexts["foreign-project"]["project_id"]
    )


def test_verification_and_preparation_required_before_start(fx):
    for action in [fx.rt.prepare, fx.rt.start]:
        with pytest.raises(RuntimeError):
            action()
    assert not fx.rt.root.exists()


def test_changed_installed_package_is_rejected(fx):
    fx.package.write_bytes(b"changed")
    with pytest.raises(ValueError, match="installed mnemos bytes"):
        fx.rt.verify()
    assert not fx.rt.root.exists()


def test_live_queue_rejection_is_actually_reached(fx, monkeypatch):
    def live(path, pids):
        assert pids == [123]
        raise ValueError("worker is live")

    monkeypatch.setattr(m, "require_prior_queue_terminal", live)
    with pytest.raises(ValueError, match="worker is live"):
        fx.rt.verify()
    assert not fx.rt.root.exists()


def test_changed_plugin_is_rejected(fx):
    (fx.source / "hooks/hooks.json").write_text("changed")
    with pytest.raises(ValueError, match="plugin source differs"):
        fx.rt.verify()


def test_hermes_acceptance_binds_executed_input_hashes(fx, monkeypatch):
    required = {}
    for relative in m.HERMES_INPUTS:
        path = fx.rt.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        required[relative] = m.sha(path)
    report = json.loads(fx.rt.hermes_acceptance.read_text())
    report["runner_commit"] = "fixture-commit"
    report["input_hashes"] = required
    monkeypatch.setattr(m, "verify_committed_inputs", lambda repo, paths, commit: None)
    m.verify_hermes_input_binding(fx.rt.repo, report)

    report["input_hashes"][m.HERMES_INPUTS[0]] = "0" * 64
    with pytest.raises(ValueError, match="input hash differs"):
        m.verify_hermes_input_binding(fx.rt.repo, report)


@pytest.mark.parametrize("mutation", ["binding", "empty", "false", "missing", "duplicate"])
def test_hermes_requires_bound_complete_phases(fx, mutation):
    p = fx.rt.hermes_acceptance
    r = json.loads(p.read_text())
    if mutation == "binding":
        r.pop("wheel_sha256")
    elif mutation == "empty":
        r["phases"][0]["gates"] = {}
    elif mutation == "false":
        r["phases"][0]["gates"] = {"observed": False}
    elif mutation == "missing":
        r["phases"].pop()
    else:
        r["phases"].append(r["phases"][0])
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="Hermes"):
        fx.rt.verify()


def test_failed_reverification_revokes_previous_success(fx):
    fx.rt.verify()
    fx.rt.checks.write_text("{}")
    with pytest.raises(ValueError):
        fx.rt.verify()
    with pytest.raises(RuntimeError, match="verify"):
        fx.rt.prepare()


def test_native_dispatch_is_traced_without_controller_call(fx, monkeypatch):
    fx.rt.verify()
    fx.rt.prepare()
    response = {"status": "committed"}

    class Dispatcher:
        def __init__(self, *args):
            pass

        async def dispatch(self, operation, payload):
            return response

    class Service:
        def __init__(self, **kwargs):
            self.dispatcher = kwargs["dispatcher_factory"]()

        def start(self):
            return {"status": "test-double"}

        def submit(self, operation, payload):
            return asyncio.run(self.dispatcher.dispatch(operation, payload))

    monkeypatch.setattr("mnemos.memory_service.LocalMemoryService", Dispatcher)
    monkeypatch.setattr("mnemos.service.MnemosService", Service)
    monkeypatch.setattr("mnemos.engine.MnemosEngine", lambda **kwargs: object())
    monkeypatch.setattr(
        "mnemos.utils.embeddings.OllamaEmbeddingProvider", lambda **kwargs: object()
    )
    monkeypatch.setattr("mnemos.utils.storage.SQLiteStore", lambda *args: object())
    fx.rt.start()
    fx.rt.set_phase("native-teach")
    payload = {"context": {"host": "codex"}, "content": "fact"}
    assert fx.rt.service.submit("remember", payload) is response
    payload["context"]["host"] = "later"
    response["status"] = "later"
    row = fx.rt.dispatch_records[0]
    assert row["payload"]["context"]["host"] == "codex"
    assert row["result"]["status"] == "committed"
    assert row["phase"] == "native-teach"
