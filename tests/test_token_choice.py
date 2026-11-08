import copy
import math

import pytest

from benchmarks.local_memory.token_choice import decode_token_choice


def response():
    return {
        "done": True,
        "response": "A",
        "logprobs": [
            {
                "token": "A",
                "logprob": math.log(0.6),
                "top_logprobs": [
                    {"token": "A", "logprob": math.log(0.6)},
                    {"token": "B", "logprob": math.log(0.2)},
                    {"token": "C", "logprob": math.log(0.1)},
                ],
            }
        ],
    }


def test_preserves_vocabulary_mass_without_renormalizing():
    raw = response()
    original = copy.deepcopy(raw)
    result = decode_token_choice(raw, ("B", "A"))
    assert result.choice == "A"
    assert dict(result.probabilities) == pytest.approx({"B": 0.2, "A": 0.6})
    assert result.remaining_mass == pytest.approx(0.2)
    assert raw == original


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "nan",
        "positive",
        "bool",
        "mass",
        "mismatch",
        "bytes",
        "thinking",
        "unfinished",
        "multiple",
        "text",
    ],
)
def test_malformed_scores_are_unavailable_to_caller(fault):
    raw = response()
    token = raw["logprobs"][0]
    if fault == "missing":
        token["top_logprobs"].pop(1)
    elif fault == "duplicate":
        token["top_logprobs"].append(token["top_logprobs"][0])
    elif fault in ("nan", "positive", "bool"):
        token["logprob"] = {"nan": float("nan"), "positive": 0.1, "bool": True}[fault]
    elif fault == "mass":
        token["top_logprobs"][2]["logprob"] = 0
    elif fault == "mismatch":
        token["logprob"] = math.log(0.5)
    elif fault == "bytes":
        token["bytes"] = [66]
    elif fault == "thinking":
        raw["thinking"] = "reasoning"
    elif fault == "unfinished":
        raw["done"] = False
    elif fault == "multiple":
        raw["logprobs"].append(copy.deepcopy(token))
    elif fault == "text":
        raw["response"] = " A"
    with pytest.raises(ValueError):
        decode_token_choice(raw, ("A", "B"))
