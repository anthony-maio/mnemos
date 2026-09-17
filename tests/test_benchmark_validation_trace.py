"""Invalid model output must remain recordable in benchmark checkpoints."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from benchmarks.local_memory import DEFAULT_PROFILES, make_ollama_service_factory
from benchmarks.local_memory.resumption import Checkpoint, save_checkpoint
from mnemos.contracts import ExtractionResponse
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry


@pytest.mark.asyncio
async def test_model_validation_failure_survives_trace_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def invalid_output(self: LocalMemoryService, job: object) -> None:
        ExtractionResponse.model_validate(
            {"decision": "no_durable_facts", "facts": [], "reason": "   "}
        )

    monkeypatch.setattr(LocalMemoryService, "process_job", invalid_output)
    factory = make_ollama_service_factory(tmp_path / "trace.db", ScopeRegistry())
    service = factory(DEFAULT_PROFILES[0], "validation-failure")
    try:
        with pytest.raises(ValidationError):
            await service.maintenance.processor(SimpleNamespace(id="job-invalid", kind="extract"))
        trace = service._benchmark_provider_trace
        assert not trace["predict_calls"]
        assert not trace["embed_calls"]
        checkpoint = Checkpoint(
            run_id="validation-failure",
            dataset_sha256="dataset",
            config_sha256="config",
            profile="full",
            repetitions=1,
            completed_observation_ids=["invalid:r1"],
            completed_observations=[{"raw_output": trace}],
        )
        path = tmp_path / "checkpoint.json"
        save_checkpoint(path, checkpoint)
        saved = json.loads(path.read_text(encoding="utf-8"))
        failure = saved["completed_observations"][0]["raw_output"]["job_exceptions"][0]
        assert failure["exception_type"] == "ValidationError"
        assert failure["kind"] == "extract"
        detail = failure["validation_errors"][0]
        assert detail["type"] == "value_error"
        assert detail["loc"] == ["reason"]
        assert detail["input"] == "   "
        assert "Reason must contain text" in detail["msg"]
        assert "Reason must contain text" in detail["ctx"]["error"]
    finally:
        service.engine.store.close()
