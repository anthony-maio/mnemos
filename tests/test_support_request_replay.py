from copy import deepcopy

import pytest
import httpx
import json

from benchmarks.local_memory.support_request_replay import grade_replays, restore_questions, run_replays
from benchmarks.local_memory.support_request_replay import direct_recorded_client
from mnemos.judgment import PREAMBLE, serialize_question


def test_frozen_questions_round_trip_without_adding_a_second_preamble():
    payload = {"support:0": {"type": "choice", "instructions": PREAMBLE + "Check the source.",
                              "criteria": {"unsupported": "No answer", "evidence:0": "Own source"}}}
    original = deepcopy(payload)
    assert {k: serialize_question(v) for k, v in restore_questions(payload).items()} == payload
    assert payload == original
    payload["support:0"]["instructions"] = "No preamble"
    with pytest.raises(ValueError, match="preamble"):
        restore_questions(payload)


@pytest.mark.asyncio
async def test_inference_refuses_gold_fields_before_provider_execution(tmp_path):
    with pytest.raises(ValueError, match="labels"):
        await run_replays([{"expected_fact_ids": ["gold"]}], "http://127.0.0.1:11439",
                          tmp_path / "invalid")


def test_unattempted_and_unavailable_cases_cannot_earn_abstention_credit():
    labels = [dict(request_id=str(i), strategy="hybrid", expected_fact_ids=[], candidate_ids=[])
              for i in range(3)]
    rows = [dict(request_id="0", status="assessed", milliseconds=1, judgment={"answers": {}}),
            dict(request_id="1", status="unavailable", milliseconds=15000)]
    result = grade_replays(rows, labels)["arms"]["hybrid"]
    assert result["questions"] == result["no_answer"] == 3
    assert result["correct_abstentions"] == result["unavailable"] == result["unattempted"] == 1
    assert result["abstention"] == 1 / 3
    assert result["precision"] is None


@pytest.mark.asyncio
async def test_transport_records_exact_request_and_response_and_rejects_remote(tmp_path):
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(200, text='{"done":true}')
    client_type = direct_recorded_client(tmp_path, lambda: "bound-request")
    async with client_type(transport=httpx.MockTransport(respond)) as client:
        await client.post("http://127.0.0.1:11439/api/chat", json={"prompt": "exact", "format": {}})
        with pytest.raises(ValueError, match="loopback"):
            await client.post("https://example.com", json={})
    assert len(calls) == 1
    request = json.loads((tmp_path / "requests.jsonl").read_text())
    response = json.loads((tmp_path / "responses.jsonl").read_text())
    assert request["request_id"] == response["request_id"] == "bound-request"
    assert request["body"] == calls[0].content.decode()
    assert response["body"] == '{"done":true}'


@pytest.mark.asyncio
async def test_transport_evidence_write_failure_aborts_request(tmp_path):
    client_type = direct_recorded_client(tmp_path / "missing-directory", lambda: "request")
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(200)
    async with client_type(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(FileNotFoundError):
            await client.post("http://127.0.0.1:11439/api/chat", json={})
    assert calls == []
