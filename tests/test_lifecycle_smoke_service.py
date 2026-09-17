from pathlib import Path
import json
from typing import Any

from benchmarks.local_memory.lifecycle_target_smoke import run_lifecycle_target_smoke
from mnemos.config import MnemosConfig, ModuleConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore


def test_development_diagnostic_observes_actual_service_correction_and_forget(
    tmp_path: Path,
) -> None:
    def build(db_path: Path, registry: ScopeRegistry, **options: Any) -> Any:
        def factory(profile: Any, run_key: str) -> LocalMemoryService:
            assert profile.name == "baseline"
            config = MnemosConfig(
                modules=ModuleConfig(
                    surprisal=False,
                    mutable_rag=False,
                    affective=False,
                    sleep=False,
                    spreading=False,
                )
            )
            return LocalMemoryService(
                MnemosEngine(
                    config=config,
                    llm=UnavailableLLMProvider(),
                    embedder=SimpleEmbeddingProvider(dim=64),
                    store=SQLiteStore(str(db_path)),
                    scope_registry=registry,
                    require_scope_context=True,
                ),
                registry,
            )

        return factory

    def metadata(endpoint: str, names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        return {name: {"name": name, "digest": "a" * 64} for name in names}

    report = run_lifecycle_target_smoke(
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        endpoint="http://unused.invalid",
        model="fixture",
        embedding_model="fixture-embedding",
        profile_names=("baseline",),
        service_factory_builder=build,
        model_metadata_loader=metadata,
    )
    assert report["status"] == "complete", [run["diagnostic"] for run in report["runs"]]
    corrected, forgotten = [run["diagnostic"] for run in report["runs"]]
    assert corrected["correct_job_processed"]
    assert corrected["current_version_observed"]
    assert corrected["restart_after_correction"]
    assert forgotten["forget_dispatched"]
    assert forgotten["replay_suppressed"]
    assert forgotten["abstain_observed"]
    raw_rows = Path(report["raw_outputs_path"]).read_text(encoding="utf-8").splitlines()
    assert len(raw_rows) == 2
    assert all(json.loads(row)["status"] == "complete" for row in raw_rows)
    manifest = json.loads(Path(report["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["completed_runs"] == 2
