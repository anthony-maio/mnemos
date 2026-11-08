"""Bounded development transport; interpretation is not verified entailment."""

import asyncio
from dataclasses import dataclass

from .compact_query_wire import TransportReply
from .condition_applicability import QueryCircumstances, evaluate_conditions
from .condition_query_wire import decode_condition_question, render_condition_question


@dataclass(frozen=True)
class ConditionResult:
    status: str
    circumstances: QueryCircumstances | None = None
    applicability: str | None = None
    raw_response: str | None = None
    error: str | None = None


def bind_condition_transport(provider):
    """Runtime owners must separately retain exact provider requests and replies."""

    async def transport(request):
        prompt, schema = render_condition_question(request)
        wire = await provider(prompt, schema)
        if (
            not isinstance(wire, dict)
            or wire.get("done") is not True
            or not isinstance(wire.get("response"), str)
        ):
            raise ValueError("Provider did not confirm complete condition text")
        return TransportReply(request.request_id, request.binding_sha256, True, wire["response"])

    return transport


async def interpret_conditions(request, transport, fence):
    """One attempt, 15-second deadline; cancellation propagates to the owner.

    An unavailable interpretation has no applicability value. A structurally
    accepted interpretation still needs independent semantic evaluation.
    """
    raw = None
    try:
        fence()
        render_condition_question(request)
        response = await asyncio.wait_for(transport(request), timeout=15)
        raw = response.text
        circumstances = decode_condition_question(request, response, fence)
        applicability = evaluate_conditions(
            request.source,
            circumstances,
            source_text=request.source_text,
            query_text=request.query,
        )
        fence()
        return ConditionResult("interpreted", circumstances, applicability, raw)
    except Exception as exc:
        return ConditionResult("unavailable", raw_response=raw, error=type(exc).__name__)
