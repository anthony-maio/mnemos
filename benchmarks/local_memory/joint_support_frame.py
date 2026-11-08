"""Model-free serialization for the inactive joint-cohort support experiment."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import hashlib
import json


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected nonempty text")
    return value


def _keys(value: object, expected: set[str]) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Unexpected inference fields; labels belong outside inputs")
    return value


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class JointFrame:
    input_ids: tuple[int, ...]
    target_ids: tuple[str, ...]
    marker_starts: tuple[int, ...]
    decision_positions: tuple[int, ...]
    payload_json: str
    tokenizer_sha256: str
    frame_sha256: str


def serialize_joint_cohort(
    request: Mapping,
    encode: Callable[[str], Sequence[int]],
    *,
    tokenizer_sha256: str,
    special_ids: frozenset[int],
    max_tokens: int,
    max_bytes: int = 65_536,
) -> JointFrame:
    """Encode all evidence, then append structurally bound decision positions.

    The caller supplies an encoder with special-token insertion disabled. No
    role template, label, predicted decision or vocabulary output is appended.
    All limits fail the entire cohort; no source or competing fact is dropped.
    """
    if (
        not isinstance(tokenizer_sha256, str)
        or len(tokenizer_sha256) != 64
        or any(c not in "0123456789abcdef" for c in tokenizer_sha256)
        or type(max_tokens) is not int
        or max_tokens < 1
        or type(max_bytes) is not int
        or max_bytes < 1
    ):
        raise ValueError("Invalid tokenizer binding or budget")
    if not isinstance(special_ids, frozenset) or any(
        type(t) is not int or t < 0 for t in special_ids
    ):
        raise ValueError("Expected explicit reserved-token inventory")
    request = _keys(request, {"query", "reference_date", "candidates"})
    query = _text(request["query"])
    reference = _text(request["reference_date"])
    if date.fromisoformat(reference).isoformat() != reference:
        raise ValueError("Expected ISO reference date")
    candidates = request["candidates"]
    if not isinstance(candidates, list) or len(candidates) > 5:
        raise ValueError("Expected zero through five candidates")
    normalized = []
    target_ids = []
    for candidate in candidates:
        candidate = _keys(candidate, {"id", "version", "content", "assertion_status", "sources"})
        target = _text(candidate["id"])
        if (
            target in target_ids
            or type(candidate["version"]) is not int
            or candidate["version"] < 1
        ):
            raise ValueError("Invalid target identity/version")
        if candidate["assertion_status"] not in {"asserted", "unknown", "conflict"}:
            raise ValueError("Unknown assertion status")
        sources = candidate["sources"]
        if not isinstance(sources, list) or len(sources) > 64:
            raise ValueError("Unsupported source inventory")
        retained = []
        source_ids = set()
        for source in sources:
            source = _keys(source, {"id", "source_event_id", "excerpt"})
            source_id = _text(source["id"])
            if source_id in source_ids:
                raise ValueError("Duplicate target source")
            source_ids.add(source_id)
            retained.append(
                {key: _text(source[key]) for key in ("id", "source_event_id", "excerpt")}
            )
        target_ids.append(target)
        normalized.append(
            dict(
                id=target,
                version=candidate["version"],
                content=_text(candidate["content"]),
                assertion_status=candidate["assertion_status"],
                sources=retained,
            )
        )
    payload = json.dumps(
        dict(query=query, reference_date=reference, candidates=normalized),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(payload.encode("utf-8")) > max_bytes:
        raise ValueError("Cohort byte budget exceeded")

    def tokens(text: str) -> tuple[int, ...]:
        result = tuple(encode(text))
        if not result or any(type(t) is not int or t < 0 or t in special_ids for t in result):
            raise ValueError("Invalid or reserved token in inference input")
        return result

    ids = list(tokens("MNEMOS_JOINT_SUPPORT_V1\n" + payload))
    starts, positions = [], []
    if len(ids) > max_tokens:
        raise ValueError("Cohort token budget exceeded")
    for index in range(len(target_ids)):
        marker = tokens(f"\nDecision target index: {index}\n")
        starts.append(len(ids))
        ids.extend(marker)
        positions.append(len(ids) - 1)
        if len(ids) > max_tokens:
            raise ValueError("Decision marker exceeds token budget")
    binding = json.dumps(
        dict(
            version=1,
            tokenizer_sha256=tokenizer_sha256,
            input_ids=ids,
            payload_sha256=_digest(payload),
            target_ids=target_ids,
            marker_starts=starts,
            decision_positions=positions,
        ),
        sort_keys=True,
    )
    return JointFrame(
        tuple(ids),
        tuple(target_ids),
        tuple(starts),
        tuple(positions),
        payload,
        tokenizer_sha256,
        _digest(binding),
    )
