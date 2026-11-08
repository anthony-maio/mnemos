"""Inventory evidence must precede memory calls and cover the complete set."""

from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.native_hosts import shared_trace as trace
from tests.test_codex_code_mode_trace import fixture, parse

EXPECTED = "mcp__mnemos__mnemos_retrieve"


def test_observation_prefix_preserves_the_frozen_lifecycle_request():
    prompt = "  Retain caf\u00e9 evidence.\nKeep the original job ID.\n"
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    observed = trace.inventory_observation_prompt(prompt, expected_sha256=digest)
    assert observed == trace.CODEX_INVENTORY_PREFIX + prompt
    assert observed.removeprefix(trace.CODEX_INVENTORY_PREFIX).encode() == prompt.encode()


@pytest.mark.parametrize("changed", ["different", "original\n", " original"])
def test_changed_lifecycle_request_is_rejected_before_observation(changed):
    digest = hashlib.sha256(b"original").hexdigest()
    with pytest.raises(ValueError, match="frozen pre-inventory request"):
        trace.inventory_observation_prompt(changed, expected_sha256=digest)


def inventory_fixture(inventories=None, *, source=None, late=False):
    events, payloads, session = fixture()
    blocks = []
    inventories = [["clock__curr_time", EXPECTED]] if inventories is None else inventories
    source = trace.CODEX_INVENTORY_SOURCE if source is None else source

    def reference(value, kind):
        number = str(len(payloads) + 1)
        payloads[number] = value
        return {
            "raw_payload_id": "raw_payload:" + number,
            "kind": {"type": kind},
            "path": "payloads/" + number + ".json",
        }

    for index, names in enumerate(inventories):
        identity = "inventory-" + str(index)
        call = {
            "type": "custom_tool_call",
            "call_id": "inventory-model-" + str(index),
            "name": "exec",
            "input": source,
        }
        result = {
            "response": {
                "Result": {
                    "cell_id": identity,
                    "content_items": [{"type": "input_text", "text": json.dumps(names)}],
                    "error_text": None,
                }
            }
        }
        blocks.extend(
            [
                {
                    "type": "inference_started",
                    "inference_call_id": "inventory-inference-" + str(index),
                    "model": "pinned",
                    "request_payload": reference({"model": "pinned"}, "inference_request"),
                },
                {
                    "type": "inference_completed",
                    "inference_call_id": "inventory-inference-" + str(index),
                    "response_id": "inventory-response-" + str(index),
                    "response_payload": reference(
                        {"response_id": "inventory-response-" + str(index), "output_items": [call]},
                        "inference_response",
                    ),
                },
                {
                    "type": "code_cell_started",
                    "runtime_cell_id": identity,
                    "model_visible_call_id": call["call_id"],
                    "source_js": source,
                },
                {
                    "type": "code_cell_initial_response",
                    "runtime_cell_id": identity,
                    "status": "completed",
                    "response_payload": reference(deepcopy(result), "tool_result"),
                },
                {
                    "type": "code_cell_ended",
                    "runtime_cell_id": identity,
                    "status": "completed",
                    "response_payload": reference(deepcopy(result), "tool_result"),
                },
            ]
        )
        session.append({"type": "response_item", "payload": deepcopy(call)})
    offset = len(events) - 3 if late else 3
    events[offset:offset] = blocks
    return events, payloads, session


def check(tmp_path, data, *, allowed=None, observe=False):
    parse(tmp_path, *data, observe=observe)
    source = tmp_path / "session.jsonl"
    provenance = trace.TraceProvenance(
        "codex", 42, "session-1", str(source), hashlib.sha256(source.read_bytes()).hexdigest()
    )
    reader = trace.observe_codex_tool_inventory if observe else trace.parse_codex_tool_inventory
    return reader(
        tmp_path / "trace",
        provenance=provenance,
        model="pinned",
        allowed_tools={"mnemos_retrieve"} if allowed is None else allowed,
    )


@pytest.mark.parametrize("initial_empty", [False, True])
@pytest.mark.parametrize("observe", [False, True])
def test_complete_inventory_precedes_correlated_memory_call(tmp_path, initial_empty, observe):
    inventories = [["clock__curr_time"], ["clock__curr_time", EXPECTED]] if initial_empty else None
    calls, hashes, receipt = check(tmp_path, inventory_fixture(inventories), observe=observe)
    assert len(calls) == 1 and calls[0].name == "mnemos_retrieve"
    assert receipt["status"] == "verified" and receipt["inventory_before_memory_calls"]
    assert receipt["expected_mnemos_names"] == [EXPECTED]
    assert len(receipt["attempts"]) == (2 if initial_empty else 1)
    for attempt in receipt["attempts"]:
        assert attempt["finished_sequence"] < receipt["first_memory_sequence"]
        assert attempt["initial_response_sha256"] in hashes.values()
        assert attempt["final_response_sha256"] in hashes.values()


@pytest.mark.parametrize(
    "inventories",
    [
        [["mcp__mnemos__mnemos_inspect", EXPECTED]],
        [["mcp__mnemos__mnemos_inspect"]],
        [[EXPECTED, EXPECTED]],
        [[EXPECTED, "clock__curr_time"]],
        [[False]],
        [[{"name": EXPECTED}]],
        [["invalid tool name", EXPECTED]],
        [[], []],
        [[], [], [EXPECTED]],
    ],
)
@pytest.mark.parametrize("observe", [False, True])
def test_wrong_or_unavailable_inventory_cannot_pass(tmp_path, inventories, observe):
    with pytest.raises(ValueError):
        check(tmp_path, inventory_fixture(inventories), observe=observe)


@pytest.mark.parametrize(
    "source",
    [
        'text(["mcp__mnemos__mnemos_retrieve"]);',
        'text(ALL_TOOLS.filter(t => t.name.includes("retrieve")).map(t => t.name).sort());',
        trace.CODEX_INVENTORY_SOURCE + " await tools.example();",
    ],
)
def test_fabricated_filtered_or_impure_source_is_rejected(tmp_path, source):
    with pytest.raises(ValueError, match="ALL_TOOLS expression"):
        check(tmp_path, inventory_fixture(source=source))


def test_catalog_after_memory_call_does_not_establish_startup_inventory(tmp_path):
    with pytest.raises(ValueError):
        check(tmp_path, inventory_fixture(late=True))


@pytest.mark.parametrize(
    "mutation",
    [
        "cell",
        "error",
        "missing-error",
        "extra-item",
        "wrong-type",
        "different",
        "failed",
        "missing-initial",
        "ambiguous",
        "bad-json",
        "not-list",
    ],
)
def test_inventory_acknowledgement_must_match_native_cell(tmp_path, mutation):
    events, payloads, session = inventory_fixture()
    event = next(
        row
        for row in events
        if row["type"] == "code_cell_initial_response" and row["runtime_cell_id"] == "inventory-0"
    )
    key = event["response_payload"]["raw_payload_id"].split(":")[1]
    result = payloads[key]["response"]["Result"]
    if mutation == "cell":
        result["cell_id"] = "other-cell"
    elif mutation == "error":
        result["error_text"] = "inventory failed"
    elif mutation == "missing-error":
        del result["error_text"]
    elif mutation == "extra-item":
        result["content_items"].append({"type": "input_text", "text": "extra"})
    elif mutation == "wrong-type":
        result["content_items"][0]["type"] = "image"
    elif mutation == "different":
        result["content_items"][0]["text"] = "[]"
    elif mutation == "failed":
        event["status"] = "failed"
    elif mutation == "missing-initial":
        events.remove(event)
    elif mutation == "ambiguous":
        payloads[key]["response"]["Error"] = "ambiguous"
    elif mutation == "bad-json":
        result["content_items"][0]["text"] = "["
    else:
        result["content_items"][0]["text"] = "{}"
    with pytest.raises(ValueError):
        check(tmp_path, (events, payloads, session))


@pytest.mark.parametrize("observe", [False, True])
def test_inventory_payload_cannot_change_after_native_trace_verification(
    tmp_path, monkeypatch, observe
):
    name = "observe_codex_code_mode_trace" if observe else "parse_codex_code_mode_trace"
    original = getattr(trace, name)

    def change_after_parse(directory, **kwargs):
        result = original(directory, **kwargs)
        for path in (directory / "payloads").iterdir():
            payload = json.loads(path.read_bytes())
            if "response" in payload:
                path.write_text("{}", encoding="utf-8")
                break
        return result

    data = inventory_fixture()
    parse(tmp_path, *data)
    monkeypatch.setattr(trace, name, change_after_parse)
    source = tmp_path / "session.jsonl"
    with pytest.raises(ValueError, match="changed after trace verification"):
        reader = trace.observe_codex_tool_inventory if observe else trace.parse_codex_tool_inventory
        reader(
            tmp_path / "trace",
            provenance=trace.TraceProvenance(
                "codex",
                42,
                "session-1",
                str(source),
                hashlib.sha256(source.read_bytes()).hexdigest(),
            ),
            model="pinned",
            allowed_tools={"mnemos_retrieve"},
        )


def test_observed_inventory_retains_unsuccessful_operation_and_native_sequences(tmp_path):
    events, payloads, session = inventory_fixture()
    result = {"status": "degraded", "error": "connection unavailable"}
    ok = payloads["5"]["result"]["Ok"]
    ok["content"][0]["text"] = json.dumps(result)
    ok["structuredContent"]["result"] = json.dumps(result)
    calls, _, inventory = check(tmp_path / "observation", (events, payloads, session), observe=True)
    assert inventory["status"] == "verified"
    assert len(calls) == 1 and calls[0].outcome == "unsuccessful"
    assert calls[0].result == result
    assert calls[0].started_sequence < calls[0].completed_sequence
    with pytest.raises(ValueError):
        check(tmp_path / "strict", (events, payloads, session))


def test_complete_twelve_tool_learning_inventory_requires_every_advertised_name(tmp_path):
    allowed = {
        "mnemos_store",
        "mnemos_retrieve",
        "mnemos_inspect",
        "mnemos_correct",
        "mnemos_forget",
        "mnemos_forget_preview",
        "mnemos_forget_related",
        "mnemos_job_status",
        "mnemos_consolidate",
        "mnemos_list",
        "mnemos_health",
        "mnemos_feedback",
    }
    names = sorted("mcp__mnemos__" + name for name in allowed)
    calls, _, inventory = check(
        tmp_path / "all", inventory_fixture([names]), allowed=allowed, observe=True
    )
    assert len(calls) == 1 and inventory["expected_mnemos_names"] == names
    with pytest.raises(ValueError):
        check(tmp_path / "partial", inventory_fixture([names[:-1]]), allowed=allowed, observe=True)


def test_wrong_allowlist_cannot_accept_used_memory_tool(tmp_path):
    with pytest.raises(ValueError, match="outside the inventory allowlist"):
        check(tmp_path, inventory_fixture(), allowed={"mnemos_inspect"})


def test_inventory_readiness_cells_cannot_overlap(tmp_path):
    events, payloads, session = inventory_fixture([[], [EXPECTED]])
    first_end = next(
        row
        for row in events
        if row["type"] == "code_cell_ended" and row["runtime_cell_id"] == "inventory-0"
    )
    events.remove(first_end)
    second_start = next(
        i
        for i, row in enumerate(events)
        if row["type"] == "code_cell_started" and row["runtime_cell_id"] == "inventory-1"
    )
    events.insert(second_start + 1, first_end)
    with pytest.raises(ValueError, match="overlapping"):
        check(tmp_path, (events, payloads, session))


def test_successful_inventory_cannot_overlap_a_later_memory_cell(tmp_path):
    events, payloads, session = inventory_fixture()
    inventory_end = next(
        row
        for row in events
        if row["type"] == "code_cell_ended" and row["runtime_cell_id"] == "inventory-0"
    )
    events.remove(inventory_end)
    memory_start = next(
        i
        for i, row in enumerate(events)
        if row["type"] == "code_cell_started" and row["runtime_cell_id"] == "cell-1"
    )
    events.insert(memory_start + 1, inventory_end)
    with pytest.raises(ValueError, match="overlapping"):
        check(tmp_path, (events, payloads, session))
