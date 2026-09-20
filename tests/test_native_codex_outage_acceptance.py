"""Synthetic gates for the bounded Codex outage observation."""

from copy import deepcopy

import pytest

from benchmarks.native_hosts.codex_service_outage_v1 import (
    ACK,
    MODEL,
    PROMPT,
    require_codex_transcript,
)


def records():
    return [
        {"type": "session_meta", "payload": {"id": "session-1"}},
        {"type": "turn_context", "payload": {"model": MODEL, "turn_id": "turn-1"}},
        {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "turn-1"}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": PROMPT}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": ACK}],
            },
        },
        {"type": "event_msg", "payload": {"type": "task_complete", "turn_id": "turn-1"}},
    ]


def test_exact_codex_prompt_ack_and_model_pass():
    require_codex_transcript(records(), prompt=PROMPT, acknowledgement=ACK, model=MODEL)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[3]["payload"].update(message="different"),
        lambda rows: rows[4]["payload"].update(
            content=[{"type": "output_text", "text": "different"}]
        ),
        lambda rows: rows[1]["payload"].update(model="other-model"),
        lambda rows: rows.insert(
            4, {"type": "response_item", "payload": {"type": "function_call"}}
        ),
        lambda rows: rows.insert(
            4, {"type": "response_item", "payload": {"type": "function_call_output"}}
        ),
        lambda rows: rows[4]["payload"].update(
            content=[{"type": "output_text", "text": ACK}, {"type": "tool_result"}]
        ),
    ],
)
def test_codex_gate_rejects_prompt_ack_model_and_tool_mutations(mutation):
    value = records()
    mutation(value)
    with pytest.raises(ValueError):
        require_codex_transcript(value, prompt=PROMPT, acknowledgement=ACK, model=MODEL)


def test_codex_gate_rejects_missing_final_answer():
    value = deepcopy(records())
    value.pop(4)
    with pytest.raises(ValueError):
        require_codex_transcript(value, prompt=PROMPT, acknowledgement=ACK, model=MODEL)


@pytest.mark.parametrize(
    "kind",
    ["tool_search_call", "tool_search_output", "mcp_tool_call_end", "custom_tool_call_output"],
)
def test_codex_gate_rejects_other_observed_tool_types(kind):
    value = records()
    value.insert(4, {"type": "response_item", "payload": {"type": kind}})
    with pytest.raises(ValueError):
        require_codex_transcript(value, prompt=PROMPT, acknowledgement=ACK, model=MODEL)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[1]["payload"].update(turn_id="another-turn"),
        lambda rows: rows[5]["payload"].update(turn_id="another-turn"),
        lambda rows: rows.pop(5),
        lambda rows: rows.pop(0),
        lambda rows: rows.insert(0, deepcopy(rows[0])),
        lambda rows: rows[4]["payload"].update(
            content=[{"type": "output_text", "text": ACK + " "}]
        ),
    ],
)
def test_codex_gate_rejects_incomplete_or_mismatched_boundaries(mutation):
    value = records()
    mutation(value)
    with pytest.raises(ValueError):
        require_codex_transcript(value, prompt=PROMPT, acknowledgement=ACK, model=MODEL)
