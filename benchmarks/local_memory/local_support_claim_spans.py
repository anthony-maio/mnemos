"""Inactive complete-context claim spans and FP32 readout; no semantic model."""

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

_name = "local_support_claim_span_learning"
_spec = importlib.util.spec_from_file_location(
    _name, Path(__file__).with_name("local_support_v2_learning.py")
)
learning = importlib.util.module_from_spec(_spec)
sys.modules[_name] = learning
_spec.loader.exec_module(learning)
PREAMBLE = "MNEMOS_CLAIM_SPAN_SUPPORT_V1\n"


@dataclass(frozen=True)
class TextSpan:
    characters: tuple[int, int]
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class OwnedSourceSpan:
    id: str
    source_event_id: str
    characters: tuple[int, int]
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class ClaimSpan:
    id: str
    version: int
    characters: tuple[int, int]
    tokens: tuple[int, ...]
    sources: tuple[OwnedSourceSpan, ...]


@dataclass(frozen=True)
class ClaimFrame:
    input_ids: tuple[int, ...]
    text: str
    payload_json: str
    query: TextSpan
    reference_date: TextSpan
    claims: tuple[ClaimSpan, ...]
    tokenizer_binding: str
    frame_sha256: str


def _serialize(value):
    """Record string-literal ranges while writing the canonical JSON structure."""
    parts, spans = [], {}
    length = len(PREAMBLE)

    def emit(text):
        nonlocal length
        parts.append(text)
        length += len(text)

    def visit(item, path):
        if isinstance(item, dict):
            emit("{")
            for i, key in enumerate(sorted(item)):
                if i:
                    emit(",")
                emit(json.dumps(key, ensure_ascii=False) + ":")
                visit(item[key], path + (key,))
            emit("}")
        elif isinstance(item, list):
            emit("[")
            for i, child in enumerate(item):
                if i:
                    emit(",")
                visit(child, path + (i,))
            emit("]")
        else:
            encoded = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            if isinstance(item, str):
                spans[path] = (length + 1, length + len(encoded) - 1)
            emit(encoded)

    visit(value, ())
    return "".join(parts), spans


def frame_claim_spans(value, tokenizer, binding, *, max_tokens=8192, max_bytes=64000):
    """Keep the whole request; include every token overlapping a complete value.

    Character ranges are half-open offsets in preamble plus JSON, excluding
    string quotes. JSON escapes stay in their serialized form. A token crossing
    a quote/value boundary is included; a token overlapping two semantic values
    makes the mapping ambiguous and fails. Native CLS/SEP have no text offsets.
    """
    if (
        type(max_tokens) is not int
        or not 2 <= max_tokens <= 8192
        or type(max_bytes) is not int
        or not 1 <= max_bytes <= 64000
    ):
        raise ValueError("Invalid bounded input budget")
    validated = learning.load_sibling("joint_support_frame").serialize_joint_cohort(
        value,
        lambda text: list(text.encode("utf-8")),
        tokenizer_sha256=binding,
        special_ids=frozenset(),
        max_tokens=1_000_000,
        max_bytes=max_bytes,
    )
    normalized = json.loads(validated.payload_json)
    if any(len(c["sources"]) > 8 for c in normalized["candidates"]):
        raise ValueError("Source inventory exceeds eight complete owned sources")
    payload, characters = _serialize(normalized)
    if payload != validated.payload_json:
        raise ValueError("Canonical payload changed during span recording")
    text = PREAMBLE + payload
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    if type(cls) is not int or type(sep) is not int or min(cls, sep) < 0 or cls == sep:
        raise ValueError("Expected distinct native boundary tokens")
    encoded = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, truncation=False, padding=False
    )
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    special = set(tokenizer.all_special_ids)
    if (
        not ids
        or len(ids) != len(offsets)
        or any(type(i) is not int or i < 0 or i in special for i in ids)
    ):
        raise ValueError("Invalid or reserved inference tokens")
    if len(ids) + 2 > max_tokens:
        raise ValueError("Complete cohort token budget exceeded")
    previous = (-1, -1)
    for pair in offsets:
        if len(pair) != 2 or any(type(v) is not int for v in pair):
            raise ValueError("Invalid tokenizer offset type")
        a, b = pair
        if not 0 <= a < b <= len(text) or a < previous[0] or b < previous[1]:
            raise ValueError("Invalid or unordered tokenizer offsets")
        previous = (a, b)
    semantic = [("query",), ("reference_date",)]
    for i, c in enumerate(normalized["candidates"]):
        semantic.append(("candidates", i, "content"))
        semantic.extend(
            ("candidates", i, "sources", j, "excerpt") for j in range(len(c["sources"]))
        )
    mappings, owners = {}, {}
    for path in semantic:
        start, end = characters[path]
        tokens, covered = [], set()
        for index, (a, b) in enumerate(offsets, 1):
            if a < end and b > start:
                if index in owners and owners[index] != path:
                    raise ValueError("Tokenizer mapping is ambiguous across semantic values")
                owners[index] = path
                tokens.append(index)
                covered.update(range(max(a, start), min(b, end)))
        if not tokens or any(i not in covered for i in range(start, end) if not text[i].isspace()):
            raise ValueError("Incomplete semantic span token coverage")
        mappings[path] = TextSpan((start, end), tuple(tokens))
    claims = []
    for i, c in enumerate(normalized["candidates"]):
        own = []
        for j, s in enumerate(c["sources"]):
            span = mappings[("candidates", i, "sources", j, "excerpt")]
            own.append(OwnedSourceSpan(s["id"], s["source_event_id"], span.characters, span.tokens))
        span = mappings[("candidates", i, "content")]
        claims.append(ClaimSpan(c["id"], c["version"], span.characters, span.tokens, tuple(own)))
    fields = dict(
        input_ids=(cls, *ids, sep),
        text=text,
        payload_json=payload,
        query=mappings[("query",)],
        reference_date=mappings[("reference_date",)],
        claims=tuple(claims),
        tokenizer_binding=binding,
    )
    temporary = ClaimFrame(**fields, frame_sha256="")
    digest = hashlib.sha256(
        json.dumps(
            dict(schema="claim-span-support-v1", **asdict(temporary)),
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return ClaimFrame(**fields, frame_sha256=digest)


def pool_claims(hidden_states, frame):
    """Average all mapped claim tokens in FP32, retaining gradient and ID order.

    This binds a readout to content. It does not prove semantic support or
    enforce evidence ownership through the contextual encoder's attention.
    The caller must supply a trusted serializer frame, not an unauthenticated
    frame reconstructed from external JSON; index checks do not authenticate it.
    """
    import torch

    if (
        hidden_states.ndim != 2
        or hidden_states.shape[0] != len(frame.input_ids)
        or not hidden_states.is_floating_point()
    ):
        raise ValueError("Expected one complete hidden-state sequence")
    if not frame.claims:
        return hidden_states.new_empty((0, hidden_states.shape[1]), dtype=torch.float32)
    rows = []
    used = set()
    for claim in frame.claims:
        indices = claim.tokens
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(
                type(i) is not int or not 0 < i < len(frame.input_ids) - 1 or i in used
                for i in indices
            )
        ):
            raise ValueError("Invalid or ambiguous claim readout indices")
        used.update(indices)
        index = torch.tensor(indices, device=hidden_states.device, dtype=torch.long)
        pooled = hidden_states.index_select(0, index).float().mean(dim=0)
        if not bool(torch.isfinite(pooled).all()):
            raise ValueError("Nonfinite pooled claim representation")
        rows.append(pooled)
    return torch.stack(rows)


def score_claims(hidden_states, frame, head):
    import torch

    pooled = pool_claims(hidden_states, frame)
    if not frame.claims:
        return pooled.new_empty((0,))
    logits = head(pooled.to(dtype=head.weight.dtype))
    if logits.shape != (len(frame.claims), 1):
        raise ValueError("Expected one shared scalar head")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Nonfinite shared-head logits")
    return logits.reshape(-1)
