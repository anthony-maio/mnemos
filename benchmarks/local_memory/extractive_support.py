"""Pure span decoder for an experimental, development-only support screen.

A literal span establishes provenance, not entailment or answer correctness.
This module neither runs a model nor changes the production retrieval path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ExtractiveSpan:
    text: str
    start: int
    end: int
    start_token: int
    end_token: int
    score: float


@dataclass(frozen=True)
class ExtractiveDecision:
    best_span: ExtractiveSpan | None
    null_score: float
    margin: float | None
    selected: bool


def decode_source_span(
    context: str,
    source_region: tuple[int, int],
    offsets: Sequence[tuple[int, int]],
    sequence_ids: Sequence[int | None],
    attention_mask: Sequence[int],
    start_logits: Sequence[float],
    end_logits: Sequence[float],
    *,
    cls_index: int,
    max_answer_tokens: int = 30,
    null_margin: float = 0.0,
) -> ExtractiveDecision:
    """Select a contiguous source span only when it strictly beats CLS.

    Scores are raw start/end logit sums, not calibrated probabilities. Ties
    between spans choose the earliest start and then earliest end. A tie with
    CLS abstains. Question, prefix, special and padding tokens cannot supply
    an answer. Malformed input raises instead of earning abstention credit.
    """
    size = len(offsets)
    if not size or any(
        len(values) != size for values in (sequence_ids, attention_mask, start_logits, end_logits)
    ):
        raise ValueError("Token arrays must have the same nonzero length")
    if type(max_answer_tokens) is not int or max_answer_tokens <= 0:
        raise ValueError("Answer length must be a positive integer")
    if type(cls_index) is not int or not 0 <= cls_index < size:
        raise ValueError("Invalid CLS index")
    if not math.isfinite(null_margin):
        raise ValueError("Null margin must be finite")
    if any(not math.isfinite(value) for value in (*start_logits, *end_logits)):
        raise ValueError("Logits must be finite")
    lo, hi = source_region
    if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= len(context):
        raise ValueError("Source region must be a nonempty range in context")
    if any(type(mask) is not int or mask not in (0, 1) for mask in attention_mask):
        raise ValueError("Attention mask must contain integer zero or one")
    if any(seq is not None and (type(seq) is not int or seq not in (0, 1)) for seq in sequence_ids):
        raise ValueError("Sequence IDs must identify question, context or special tokens")
    if sequence_ids[cls_index] is not None or attention_mask[cls_index] != 1:
        raise ValueError("CLS must be an attended special token")
    eligible = []
    previous = (0, 0)
    for index, (start, end) in enumerate(offsets):
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            raise ValueError("Offsets must be nonnegative ordered integers")
        if sequence_ids[index] == 1 and attention_mask[index]:
            if end > len(context) or start < previous[0] or end < previous[1]:
                raise ValueError("Context offsets must be ordered and within context")
            previous = (start, end)
        eligible.append(
            sequence_ids[index] == 1 and attention_mask[index] == 1 and lo <= start < end <= hi
        )
    null_score = float(start_logits[cls_index] + end_logits[cls_index])
    if not math.isfinite(null_score):
        raise ValueError("Null score overflow")
    best: ExtractiveSpan | None = None
    for start_token in range(size):
        if not eligible[start_token]:
            continue
        for end_token in range(start_token, min(size, start_token + max_answer_tokens)):
            if not eligible[end_token]:
                break
            start, end = offsets[start_token][0], offsets[end_token][1]
            text = context[start:end]
            if not text.strip():
                continue
            score = float(start_logits[start_token] + end_logits[end_token])
            if not math.isfinite(score):
                raise ValueError("Span score overflow")
            if best is None or score > best.score:
                best = ExtractiveSpan(text, start, end, start_token, end_token, score)
    margin = None if best is None else best.score - null_score
    if margin is not None and not math.isfinite(margin):
        raise ValueError("Margin overflow")
    return ExtractiveDecision(best, null_score, margin, margin is not None and margin > null_margin)


def span_matches_fact(decision: ExtractiveDecision, content: str) -> bool:
    """A necessary literal-overlap guard, never a semantic-support guarantee."""
    return bool(
        decision.selected and decision.best_span is not None and decision.best_span.text in content
    )
