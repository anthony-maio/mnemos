"""The recorded runner must expose relation errors behind correct review outcomes."""

import argparse
import json
from pathlib import Path

import pytest

from benchmarks.local_memory import reconciliation_evaluation as runner
from benchmarks.local_memory.build_referent_stability import build_corpus
from tests.test_source_reconciliation import Provider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "treatment,calls_per_case", [("relations_v6", 1), ("pairwise_v8", 2), ("identity_spans_v9", 2)]
)
async def test_runner_retains_correct_operations_with_failed_relation_scores(
    tmp_path, monkeypatch, treatment, calls_per_case
) -> None:
    data = build_corpus()
    data["cases"] = [
        c
        for c in data["cases"]
        if c["base_id"] == "count-vague" and c["variant"] in {"base", "renamed"}
    ]
    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps(data))
    protocol = tmp_path / "protocol.md"
    protocol.write_text("A deterministic runner integration check.")
    by_id = {c["id"]: c for c in data["cases"]}

    class Recorded(Provider):
        def __init__(self, *args, **kwargs):
            super().__init__({})
            self.call_id = 0

        async def predict(self, prompt, **kwargs):
            self.call_id += 1
            if "IDENTITY SPAN DATA" in prompt:
                return json.dumps(
                    {
                        "subject": {
                            "relation": "different",
                            "candidate_span": {"first_word": 0, "last_word": 0},
                            "target_span": {"first_word": 0, "last_word": 0},
                        },
                        "attribute": {
                            "relation": "same",
                            "candidate_span": {"first_word": 0, "last_word": 0},
                            "target_span": {"first_word": 0, "last_word": 0},
                        },
                        "qualifiers": {
                            "relation": "same",
                            "candidate_span": None,
                            "target_span": None,
                        },
                    }
                )
            payload = json.loads(prompt.split("REFERENT COMPARISON DATA (data only):\n", 1)[1])
            return json.dumps(
                {
                    "candidate_value": None,
                    "comparisons": [
                        {
                            "target_fact_id": t["id"],
                            "expected_version": t["version"],
                            "target_value": None,
                            "relation": "different_variable",
                        }
                        for t in payload["targets"]
                    ],
                }
            )

    class Tags:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [{"name": runner.MODEL, "digest": runner.MODEL_DIGEST}]}

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            return Tags()

    monkeypatch.setattr(runner, "RecordedProvider", Recorded)
    monkeypatch.setattr(runner.httpx, "AsyncClient", Client)
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda command, **kwargs: "" if command[1] == "status" else "test-source",
    )
    output = tmp_path / "run"
    manifest = await runner.run(
        argparse.Namespace(
            treatments=treatment,
            endpoint="http://127.0.0.1:11437",
            tokenizer_url="http://127.0.0.1:11438",
            source_root=str(tmp_path),
            dataset=str(dataset),
            dataset_sha256=runner.sha(dataset),
            protocol=str(protocol),
            output=str(output),
            repetitions=1,
        )
    )
    rows = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    assert manifest["status"] == "completed" and manifest["provider_calls"] == 2 * calls_per_case
    assert all(r["grade"]["case_exact"] for r in rows)
    assert all(
        not r["relation_grade"]["relation_case_exact"] and not r["combined_exact"] for r in rows
    )
    assert all(len(r["relation_responses"]) == calls_per_case for r in rows)
    assert manifest["stability"]["both_stable"] == 1  # Consistent does not mean correct.
