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
from assertion_inventory import inventory_qualified_assertions_v3 as inventory_assertions, retained_qualified_assertions_v3 as retained_assertions
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
        row = {"case_id": case["id"], "trace": provider.trace}
        try:
            events = [EpisodicEvent.model_validate(case["source"])]
            inventory = await inventory_assertions(events, provider)
            retained = retained_assertions(inventory, events)
            row.update(status="runtime_valid", inventory=inventory.model_dump(mode="json"),
                retained=[a.model_dump(mode="json") for a in retained])
        except Exception as exc:
            row.update(status="error", error_type=type(exc).__name__, error=str(exc))
            state["errors"] += 1
        with (root / "observations.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        state["processed"] += 1
        save()
        print(json.dumps({k: row[k] for k in ("case_id", "status")}), flush=True)
    verify()
    state.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat(),
        observations_sha256=sha(root / "observations.jsonl"))
    save()

if __name__ == "__main__":
    parent = Path("C:/Users/Ant/AppData/Local/Temp/mnemos-assertion-inventory-development-v3")
    assert sha(parent / "preparation.json") == "b1df7daae61fb5eaa3d2e143e868e101265dd69afd78186a2f6e8276167367e5"
    parent_state = json.loads((parent / "status.json").read_bytes())
    if parent_state["status"] != "completed" or parent_state["processed"] != 46:
        raise RuntimeError("Parent v3 experiment must complete all 46 cases before v4 generation")
    assert parent_state["observations_sha256"] == sha(parent / "observations.jsonl")
    (root / "parent-terminal-binding.json").write_text(json.dumps({
        "preparation_sha256": sha(parent / "preparation.json"),
        "observations_sha256": sha(parent / "observations.jsonl"),
        "status_sha256": sha(parent / "status.json")}, indent=2) + "\n")
    asyncio.run(main())
