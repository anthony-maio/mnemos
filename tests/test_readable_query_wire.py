from dataclasses import replace
from itertools import product

import pytest

from benchmarks.local_memory.compact_query import evaluate_reply
from benchmarks.local_memory.readable_query_wire import (
    CONDITIONS,
    INTENTS,
    RELATIONS,
    evaluate_readable,
    render_readable_request,
)
from tests.test_compact_query_wire import request, reply


def body(codes=("MA", "UU"), intent="S"):
    relations = {v: k for k, v in RELATIONS.items()}
    conditions = {v: k for k, v in CONDITIONS.items()}
    return {
        "intent": {v: k for k, v in INTENTS.items()}[intent],
        "date": "2026-11-08" if intent == "D" else None,
        "judgments": {
            f"c{i}": {"relation": relations[code[0]], "conditions": conditions[code[1]]}
            for i, code in enumerate(codes)
        },
    }


@pytest.mark.asyncio
async def test_all_judgments_and_temporal_states_preserve_decisions():
    req = await request()
    codes = [r + c for r in "MXU" for c in "AXU"]
    for first, second, intent in product(codes, codes, "SDU"):
        data = body((first, second), intent)
        wire = reply(req, data)
        readable = evaluate_readable(req, wire, lambda: None)
        compact = evaluate_reply(
            req,
            reply(req, dict(t=intent, d=data["date"], c={"c0": first, "c1": second})),
            lambda: None,
        )
        assert readable.decision == compact.decision
        assert readable.selected_bases == compact.selected_bases
        assert readable.raw_response == wire.text


@pytest.mark.asyncio
async def test_full_input_and_named_judgments_survive_rendering_and_reply_order():
    req = await request()
    prompt, schema = render_readable_request(req)
    assert prompt.split("\nINPUT\n", 1)[1] == req.model_input
    assert schema["properties"]["judgments"]["required"] == [a for a, _ in req.aliases]
    data = body()
    expected = evaluate_readable(req, reply(req, data), lambda: None)
    data["judgments"] = dict(reversed(list(data["judgments"].items())))
    actual = evaluate_readable(req, reply(req, data), lambda: None)
    assert actual.decision == expected.decision
    assert actual.selected_bases == expected.selected_bases


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    ["missing", "foreign", "extra", "relation", "condition", "intent", "date", "fields", "type"],
)
async def test_invalid_judgments_are_rejected(mutation):
    req = await request()
    data = body()
    if mutation == "missing":
        del data["judgments"]["c1"]
    if mutation == "foreign":
        data["judgments"]["other"] = data["judgments"].pop("c1")
    if mutation == "extra":
        data["answer"] = "ignored"
    if mutation == "relation":
        data["judgments"]["c0"]["relation"] = "yes"
    if mutation == "condition":
        data["judgments"]["c0"]["conditions"] = True
    if mutation == "intent":
        data["intent"] = []
    if mutation == "date":
        data["date"] = "2026-11-08"
    if mutation == "fields":
        data["judgments"]["c0"]["explanation"] = "ignored"
    if mutation == "type":
        data["judgments"] = []
    with pytest.raises(ValueError):
        evaluate_readable(req, reply(req, data), lambda: None)


@pytest.mark.asyncio
async def test_duplicates_cross_request_and_incomplete_transport_are_rejected():
    req, other = await request(), await request()
    wire = reply(req, body())
    duplicate = wire.text.replace(
        '"relation": "matches"', '"relation": "matches", "relation": "excluded"'
    )
    for invalid in [
        replace(wire, text=duplicate),
        replace(wire, completed=False),
        replace(wire, request_id=other.request_id),
    ]:
        with pytest.raises(ValueError):
            evaluate_readable(req, invalid, lambda: None)
    with pytest.raises(ValueError):
        evaluate_readable(replace(req, model_input="changed"), wire, lambda: None)


@pytest.mark.asyncio
async def test_source_revocation_and_oversize_input_fail_closed():
    req = await request()

    def revoked():
        raise PermissionError("source forgotten")

    with pytest.raises(PermissionError):
        evaluate_readable(req, reply(req, body()), revoked)
    with pytest.raises(ValueError):
        evaluate_readable(req, replace(reply(req), text="x" * 64001), lambda: None)
    with pytest.raises(ValueError):
        render_readable_request(replace(req, model_input="x" * 64001))
