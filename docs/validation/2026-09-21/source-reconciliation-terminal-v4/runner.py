from pathlib import Path
from datetime import datetime, timezone
import asyncio, hashlib, json, os, subprocess, sys
from urllib.request import urlopen

root = Path(__file__).resolve().parent
inputs = json.loads((root / "inputs.json").read_bytes())
prep = json.loads((root / "preparation.json").read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
snapshot = Path(inputs["snapshot"])
if __name__ == "__main__":
    if len(sys.argv) != 2 or sha(root / "preparation.json") != sys.argv[1]:
        raise ValueError("An externally supplied matching preparation digest is required")
def verify():
    for name, digest in prep["files"].items():
        assert sha(root / name) == digest, name
    for name, digest in prep["source_files"].items():
        assert sha(snapshot / name) == digest, name
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=snapshot, text=True).strip() == prep["source_commit"]
    with urlopen(inputs["endpoint"] + "/api/tags", timeout=10) as response:
        model = next(m for m in json.loads(response.read())["models"] if m["name"] == inputs["model"])
    assert model["digest"] == prep["model"]["digest"]
verify()
sys.path.insert(0, str(snapshot))
from candidate.source_reconciliation_examples import reconcile_with_examples as reconcile_source_candidates
from materializer import materialize
from dataclasses import asdict
from mnemos.types import EpisodicEvent
from mnemos.utils.llm import LLMProvider, OllamaProvider
from mnemos.runtime import build_llm_from_settings
from mnemos.settings import AppSettings

class Provider(LLMProvider):
    supports_structured_output = True
    def __init__(self):
        self.trace = {"response_metadata": []}
        settings = AppSettings.model_validate({"llm": {"provider": "ollama", "model": inputs["model"]},
            "providers": {"ollama": {"base_url": inputs["endpoint"], "think": inputs["think"],
                "temperature": inputs["temperature"], "max_tokens": inputs["max_tokens"], "timeout": inputs["timeout"]}}})
        self.provider = build_llm_from_settings(settings)
        assert isinstance(self.provider, OllamaProvider)
        self.provider.response_observer = self.trace["response_metadata"].append
        self.trace["provider_configuration"] = settings.providers.ollama.model_dump()
    async def predict(self, prompt, **kwargs):
        self.trace.update(prompt=prompt, options=kwargs)
        result = await self.provider.predict(prompt, **kwargs)
        self.trace["response"] = result
        return result
    async def classify(self, prompt, labels):
        raise AssertionError("No classifier in inventory experiment")

async def main():
    assert not (root / "status.json").exists()
    state = {"status": "running", "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
        "preparation_sha256": sha(root / "preparation.json"), "processed": 0, "errors": 0}
    def save():
        (root / "status.json").write_text(json.dumps(state, indent=2) + "\n")
    save()
    for case in inputs["cases"]:
        provider = Provider()
        row = {"case_id": case["case_id"], "expected": case["expected"], "trace": provider.trace}
        try:
            candidates, events, targets = materialize(case)
            staged = await reconcile_source_candidates(candidates, events, targets, provider)
            row.update(status="runtime_valid", staged=[a.model_dump(mode="json") for a in staged])
        except Exception as exc:
            row.update(status="error", error_type=type(exc).__name__, error=str(exc))
            state["errors"] += 1
        row["provider_invoked"] = "prompt" in provider.trace
        with (root / "observations.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        state["processed"] += 1
        save()
        print(json.dumps({k: row[k] for k in ("case_id", "status", "provider_invoked")}), flush=True)
    verify()
    state.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat(),
        observations_sha256=sha(root / "observations.jsonl"))
    save()

if __name__ == "__main__":
    asyncio.run(main())
