"""One-pass development execution over frozen requests; never reads labels."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import date
import hashlib
import time
from typing import Any

from mnemos.answer_support import SupportCandidate, SupportEvidence
from mnemos.support_intersection import SupportSnapshot
from .relational_support_producer import prepare, produce
from .token_support_prompt import render_raw_prompt


def restore_snapshot(row: Mapping[str, Any]) -> SupportSnapshot:
    raw = row['snapshot']
    candidates = tuple(SupportCandidate(**{
        **item,
        'evidence': tuple(SupportEvidence(**e) for e in item['evidence']),
    }) for item in raw['candidates'])
    return SupportSnapshot(raw['query'], date.fromisoformat(raw['reference_date']),
                           candidates, raw['fingerprint'])


def check_requests(rows: Sequence[Mapping[str, Any]]) -> None:
    keys = [r['key'] for r in rows]
    if keys != sorted(set(keys)):
        raise ValueError('Requests must have unique keys in frozen lexical order')
    for row in rows:
        snapshot = restore_snapshot(row)
        prompt, fingerprint = prepare(snapshot, row['policy_identity'])
        if (prompt != row['prompt'] or fingerprint != row['request_fingerprint']
                or render_raw_prompt(prompt) != row['raw_prompt']
                or [c.fact_id for c in snapshot.candidates] != row['candidate_ids']
                or snapshot.fingerprint != row['snapshot_fingerprint']):
            raise ValueError('Prepared request changed')


async def run_screen(
    rows: Sequence[Mapping[str, Any]],
    transport: Callable[[str], Awaitable[dict[str, Any]]],
    emit: Callable[[str, dict[str, Any]], None],
    fence: Callable[[], None],
) -> dict[str, Any]:
    """Transport must use frozen policy and return the complete provider reply.

    The owner performs warmup and token/runtime verification before this call.
    A transport failure stops the run because completion may be uncertain.
    Completed but malformed semantic replies are retained without retry.
    """
    check_requests(rows)  # Check every request before making the first call.
    fence()
    attempted = []
    stopped = False
    for row in rows:
        started = time.perf_counter_ns()
        snapshot = restore_snapshot(row)
        uncertain = False
        completed = False
        emit('observation_start', {'key': row['key'], 'request_fingerprint': row['request_fingerprint']})

        async def provider(prompt: str) -> str:
            nonlocal uncertain, completed
            raw_prompt = render_raw_prompt(prompt)
            if raw_prompt != row['raw_prompt']:
                raise ValueError('Execution prompt changed')
            emit('request_start', {'key': row['key'], 'raw_prompt_sha256': hashlib.sha256(raw_prompt.encode()).hexdigest()})
            try:
                wire = await transport(raw_prompt)
                completed = isinstance(wire, dict) and wire.get('done') is True
                emit('provider_reply', {'key': row['key'], 'reply': wire})
                if not completed:
                    raise ValueError('Provider completion was not confirmed')
                if not isinstance(wire.get('response'), str):
                    raise ValueError('Provider did not return text')
                return wire['response']
            except BaseException:
                uncertain = not completed
                raise

        result = await produce(snapshot, row['policy_identity'], provider, fence)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        attempted.append(row['key'])
        emit('observation', {'key': row['key'], **asdict(result), 'elapsed_ms': elapsed,
                             'completion_confirmed': completed, 'completion_uncertain': uncertain})
        if uncertain:
            stopped = True
            break
        fence()
    return {'status': 'stopped_uncertain_completion' if stopped else 'completed',
            'attempted': attempted, 'unattempted': [r['key'] for r in rows if r['key'] not in attempted]}
