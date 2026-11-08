"""Development transport for explicit query judgments, with no retries."""

import asyncio

from .compact_query_wire import TransportReply, prepare_compact
from .readable_query_wire import evaluate_readable, render_readable_request
from .relation_query_v2 import QueryResult
from .relational_support import Decision


def bind_readable_transport(provider):
    """Bind replies to this invocation; the runtime owner must retain full wire logs."""

    async def transport(request):
        prompt, schema = render_readable_request(request)
        wire = await provider(prompt, schema)
        if (
            not isinstance(wire, dict)
            or wire.get("done") is not True
            or not isinstance(wire.get("response"), str)
        ):
            raise ValueError("Provider did not confirm complete readable text")
        return TransportReply(request.request_id, request.binding_sha256, True, wire["response"])

    return transport


async def interpret_readable(snapshot, source_identity, query_identity, cache, transport, fence):
    raw = None
    try:
        cohort = cache.annotate(snapshot, source_identity, fence)
        request = prepare_compact(snapshot, cohort, source_identity, query_identity, fence)
        render_readable_request(request)
        fence()
        response = await asyncio.wait_for(transport(request), timeout=15)
        fence()
        raw = response.text
        return evaluate_readable(request, response, fence)
    except Exception as exc:
        return QueryResult(Decision("unavailable"), raw, type(exc).__name__)
