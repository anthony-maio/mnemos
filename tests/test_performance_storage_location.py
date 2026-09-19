"""Performance reports bind their actual storage location without reusing data."""

import inspect
from pathlib import Path

import pytest

from benchmarks.local_memory import performance


def stub_measurements(monkeypatch):
    calls = []

    def provenance(*args):
        calls.append("provider")
        return {"fixture": True}

    def retrieval(root, **kwargs):
        calls.append(root)
        (root / "retrieval.sqlite").write_bytes(b"disposable fixture")
        return {"fixture": True}

    def capture(root):
        calls.append(root)
        (root / "capture.sqlite").write_bytes(b"disposable fixture")
        return {"fixture": True, "cold_cli_hook": {"fixture": True}}

    monkeypatch.setattr(performance, "_ollama_provenance", provenance)
    monkeypatch.setattr(performance, "measure_retrieval", retrieval)
    monkeypatch.setattr(performance, "measure_capture", capture)
    return calls


def run_kwargs(tmp_path):
    return dict(
        output=tmp_path / "report.json",
        endpoint="http://offline.invalid",
        embedding_model="fixture",
        cognition_model="fixture",
    )


def test_default_report_records_actual_temporary_storage_and_cleanup(tmp_path, monkeypatch):
    calls = stub_measurements(monkeypatch)
    report = performance.run(**run_kwargs(tmp_path))
    assert "storage_location" in report["environment"], "Timing must identify its storage volume"
    location = report["environment"]["storage_location"]
    root = calls[1]
    assert location["work_dir"] == str(root.resolve())
    assert location["selection"] == "system_temp"
    assert location["volume_anchor"] == root.anchor
    assert isinstance(location["device_id"], int)
    assert location["capture_database"] == str(root / "capture.sqlite")
    assert location["retrieval_database"] == str(root / "retrieval.sqlite")
    assert calls[2] == root
    assert not root.exists()


def test_explicit_work_directory_retains_fixture_and_records_device(tmp_path, monkeypatch):
    calls = stub_measurements(monkeypatch)
    assert "work_dir" in inspect.signature(performance.run).parameters
    root = tmp_path / "selected-volume" / "fresh-run"
    report = performance.run(**run_kwargs(tmp_path), work_dir=root)
    location = report["environment"]["storage_location"]
    assert calls[1:] == [root.resolve(), root.resolve()]
    assert location["work_dir"] == str(root.resolve())
    assert location["selection"] == "explicit"
    assert location["device_id"] == root.stat().st_dev
    assert location["cleanup"] == "retained"
    assert (root / "capture.sqlite").read_bytes() == b"disposable fixture"


@pytest.mark.parametrize("nonempty", [False, True])
def test_existing_work_directory_rejected_before_provider_calls(tmp_path, monkeypatch, nonempty):
    calls = stub_measurements(monkeypatch)
    assert "work_dir" in inspect.signature(performance.run).parameters
    root = tmp_path / "existing"
    root.mkdir()
    if nonempty:
        (root / "capture.sqlite").write_bytes(b"preserve existing data")
    with pytest.raises(FileExistsError):
        performance.run(**run_kwargs(tmp_path), work_dir=root)
    assert calls == []
    assert root.exists()
    if nonempty:
        assert (root / "capture.sqlite").read_bytes() == b"preserve existing data"


def test_failed_explicit_measurement_retains_work_for_inspection(tmp_path, monkeypatch):
    stub_measurements(monkeypatch)
    assert "work_dir" in inspect.signature(performance.run).parameters
    root = tmp_path / "failed-run"

    def fail(root, **kwargs):
        (root / "partial.sqlite").write_bytes(b"partial diagnostic")
        raise RuntimeError("fixture measurement failure")

    monkeypatch.setattr(performance, "measure_retrieval", fail)
    with pytest.raises(RuntimeError, match="fixture measurement failure"):
        performance.run(**run_kwargs(tmp_path), work_dir=root)
    assert (root / "partial.sqlite").read_bytes() == b"partial diagnostic"


def test_cli_selects_explicit_storage_directory(tmp_path, monkeypatch):
    calls = stub_measurements(monkeypatch)
    root = tmp_path / "cli-run"
    result = performance.main(["--output", str(tmp_path / "report.json"), "--work-dir", str(root)])
    assert result == 0
    assert calls[1:] == [root.resolve(), root.resolve()]


def test_existing_report_rejected_before_measurement(tmp_path, monkeypatch):
    calls = stub_measurements(monkeypatch)
    kwargs = run_kwargs(tmp_path)
    kwargs["output"].write_bytes(b"preserve prior evidence")
    with pytest.raises(FileExistsError):
        performance.run(**kwargs)
    assert calls == []
    assert kwargs["output"].read_bytes() == b"preserve prior evidence"


def test_cli_failure_does_not_overwrite_existing_report(tmp_path, monkeypatch):
    calls = stub_measurements(monkeypatch)
    output = tmp_path / "report.json"
    output.write_bytes(b"preserve prior evidence")
    assert performance.main(["--output", str(output)]) == 2
    assert output.read_bytes() == b"preserve prior evidence"
    assert calls == []
