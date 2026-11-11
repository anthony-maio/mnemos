import json

import httpx
import pytest

from benchmarks.local_memory.jev_support_replay import RecordedJevJudge, run_jev_replays
from mnemos.judgment import PREAMBLE
from mnemos.judgment import JudgmentContractError
from benchmarks.local_memory.support_request_replay import restore_questions


def _row(request_id="first"):
    return dict(
        request_id=request_id,
        purpose="support",
        state={"source": "Ada owns it."},
        questions={
            "support:0": {
                "type": "choice",
                "instructions": PREAMBLE + "Who owns it?",
                "criteria": {"evidence:0": "Ada", "unsupported": "Unknown"},
            }
        },
    )


@pytest.mark.asyncio
async def test_labels_are_rejected_before_network_or_output(tmp_path):
    calls = []
    with pytest.raises(ValueError, match="labels"):
        await run_jev_replays(
            [dict(_row(), expected_fact_ids=["gold"])],
            "secret-value",
            tmp_path / "run",
            transport=httpx.MockTransport(calls.append),
        )
    assert not calls and not (tmp_path / "run").exists()


@pytest.mark.asyncio
async def test_exact_wire_evidence_pinned_model_and_no_retry_on_failure(tmp_path):
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 3:
            return httpx.Response(503, json={"error": "unavailable"})
        assert request.headers["Authorization"] == "Bearer secret-value"
        return httpx.Response(
            200,
            json={
                "model": "jev-test-version",
                "usage": {"input_tokens": 30, "output_tokens": 2},
                "answers": {
                    key: {
                        "type": "choice",
                        "choice": next(iter(q["criteria"])),
                        "probabilities": {k: float(i == 0) for i, k in enumerate(q["criteria"])},
                        "confidence": 1.0,
                    }
                    for key, q in body["questions"].items()
                },
            },
        )

    output = tmp_path / "run"
    result = await run_jev_replays(
        [_row(str(i)) for i in range(3)],
        "secret-value",
        output,
        transport=httpx.MockTransport(respond),
    )
    assert result["completed"] == 1 and result["attempted"] == 2 and result["total"] == 3
    assert len(calls) == 3  # Warmup, success, one failure. No retries or later requests.
    assert calls[0]["model"] == "jev-latest"
    assert calls[1]["model"] == calls[2]["model"] == "jev-test-version"
    assert calls[1]["state"] == _row()["state"]
    assert calls[1]["questions"] == _row()["questions"]
    evidence = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert [json.loads(row["body"]) for row in evidence] == calls
    for path in output.iterdir():
        assert "secret-value" not in path.read_text()
    with pytest.raises(FileExistsError):
        await run_jev_replays([_row()], "secret-value", output)


@pytest.mark.asyncio
async def test_endpoint_and_recording_failures_abort_before_sending(tmp_path):
    calls = []
    judge = RecordedJevJudge(
        "secret-value", tmp_path / "missing", transport=httpx.MockTransport(calls.append)
    )
    async with judge._client() as client:
        with pytest.raises(ValueError, match="endpoint"):
            await client.post("https://example.com", json={})
        with pytest.raises(FileNotFoundError):
            await client.post("https://api.typesafe.ai/v1/systemone", json={})
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "distribution,chosen",
    [
        (None, "evidence:0"),
        ({"evidence:0": 1.0}, "evidence:0"),
        ({"evidence:0": 0.8, "unsupported": 0.8}, "evidence:0"),
        ({"evidence:0": 0.2, "unsupported": 0.8}, "evidence:0"),
        ({"evidence:0": True, "unsupported": 0.0}, "evidence:0"),
    ],
)
async def test_invalid_raw_probabilities_cannot_become_supported_answers(
    tmp_path, distribution, chosen
):
    def respond(request):
        return httpx.Response(
            200,
            json={
                "model": "jev-test",
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "answers": {
                    "support:0": {"type": "choice", "choice": chosen, "probabilities": distribution}
                },
            },
        )

    judge = RecordedJevJudge("secret-value", tmp_path, transport=httpx.MockTransport(respond))
    with pytest.raises(JudgmentContractError):
        await judge.ask(_row()["state"], restore_questions(_row()["questions"]), purpose="test")
