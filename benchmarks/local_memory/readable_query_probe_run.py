"""Serial supplied-annotation probe with durable owner-side request records."""

import asyncio
from dataclasses import asdict
from time import perf_counter

from .compact_query_wire import TransportReply
from .readable_query_wire import evaluate_readable, render_readable_request


async def run_readable_probe(requests, provider, emit, fence):
    """The provider sees prompt/schema only; grading is an offline operation.

    emit must durably record each event or raise. An uncertain provider outcome
    stops this run. Completed malformed replies are recorded without retry.
    The caller owns runtime identity checks, input token checks and cleanup.
    """
    keys = [key for key, _ in requests]
    if not keys or keys != sorted(set(keys)) or len(keys) > 64:
        raise ValueError("Expected 1-64 unique requests in frozen lexical order")
    prepared = [(key, request, *render_readable_request(request)) for key, request in requests]
    emit("run_start", dict(kind="supplied-annotation-query-only", keys=keys))
    observations = []
    for key, request, prompt, schema in prepared:
        fence()
        emit(
            "request_start",
            dict(key=key, owner_request=asdict(request), prompt=prompt, schema=schema),
        )
        started = perf_counter()
        try:
            wire = await asyncio.wait_for(provider(prompt, schema), timeout=15)
        except asyncio.CancelledError:
            emit("request_cancelled", dict(key=key, elapsed_ms=(perf_counter() - started) * 1000))
            raise
        except Exception as exc:
            emit(
                "request_uncertain",
                dict(
                    key=key,
                    error_type=type(exc).__name__,
                    elapsed_ms=(perf_counter() - started) * 1000,
                ),
            )
            emit(
                "run_stopped",
                dict(reason="uncertain_provider_completion", attempted=len(observations) + 1),
            )
            return observations
        provider_ms = (perf_counter() - started) * 1000
        # Logging failure must propagate before another provider request.
        emit("provider_reply", dict(key=key, wire=wire, provider_ms=provider_ms))
        if (
            not isinstance(wire, dict)
            or wire.get("done") is not True
            or not isinstance(wire.get("response"), str)
        ):
            emit(
                "run_stopped",
                dict(reason="unconfirmed_provider_completion", attempted=len(observations) + 1),
            )
            return observations
        fence()
        try:
            result = evaluate_readable(
                request,
                TransportReply(request.request_id, request.binding_sha256, True, wire["response"]),
                fence,
            )
            observation = dict(key=key, result=asdict(result), protocol_valid=True)
        except (ValueError, TypeError, KeyError) as exc:
            observation = dict(key=key, protocol_valid=False, error_type=type(exc).__name__)
        fence()
        observation.update(provider_ms=provider_ms, processing_ms=(perf_counter() - started) * 1000)
        emit("query_observation", observation)
        observations.append(observation)
    emit("run_complete", dict(attempted=len(observations)))
    return observations
