"""Tokenizer-only boundary controls; no semantic predictions or timing claims."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

_name = "local_support_v2_standalone_admission"
_spec = importlib.util.spec_from_file_location(
    _name, Path(__file__).with_name("local_support_v2_admission.py")
)
admission = importlib.util.module_from_spec(_spec)
sys.modules[_name] = admission
_spec.loader.exec_module(admission)
capacity = admission.capacity
frame_local_support = admission.frame_local_support

ROOT = Path(__file__).resolve().parents[2]
CONTROL_IDS = (
    "empty",
    "one-fact",
    "four-facts",
    "five-facts",
    "eight-owned-sources",
    "nine-owned-sources",
    "exact-token-bound",
    "token-overflow",
    "exact-byte-bound",
    "byte-overflow",
    "six-facts",
)


def account_controls(cases, outcomes):
    cases_by_id = {case["id"]: case for case in cases}
    observed = {row["id"]: row for row in outcomes}
    if len(cases_by_id) != len(cases) or len(observed) != len(outcomes):
        raise ValueError("Duplicate control or observation")
    if set(cases_by_id) != set(observed):
        raise ValueError("Expected complete observations for all controls")
    if any(row["availability"] not in {"admitted", "unavailable"} for row in outcomes):
        raise ValueError("Unknown availability observation")
    for case in cases:
        gold = case["supported_target_ids"]
        if not isinstance(gold, list) or len(gold) != len(set(gold)):
            raise ValueError("Expected unique supported target identities")
    unavailable = [case for case in cases if observed[case["id"]]["availability"] == "unavailable"]
    return dict(
        controls=len(cases),
        answerable_controls=sum(bool(case["supported_target_ids"]) for case in cases),
        unavailable_answerable_controls=sum(
            bool(case["supported_target_ids"]) for case in unavailable
        ),
        supported_target_opportunities=sum(len(case["supported_target_ids"]) for case in cases),
        unavailable_supported_target_opportunities=sum(
            len(case["supported_target_ids"]) for case in unavailable
        ),
        semantic_recall=None,
        semantic_precision=None,
        limits="Availability accounting only. No model decisions, semantic abstention, critical retention or fresh acceptance were measured.",
    )


def request(count=1, sources=1):
    candidates = []
    for index in range(count):
        identity = hashlib.sha256(f"boundary-fixture-{index}".encode()).hexdigest()[:24]
        content = f"Boundary fixture {index} exports JSON."
        candidates.append(
            dict(
                id=identity,
                version=1,
                content=content,
                assertion_status="asserted",
                sources=[
                    dict(
                        id=hashlib.sha256(f"{identity}-source-{n}".encode()).hexdigest()[:24],
                        source_event_id=hashlib.sha256(
                            f"{identity}-event-{n}".encode()
                        ).hexdigest()[:24],
                        excerpt=f"Retained configuration observation {n} for boundary fixture {index}: export_format=JSON.",
                    )
                    for n in range(sources)
                ],
            )
        )
    return dict(
        query="List the recorded export format for every listed boundary fixture.",
        reference_date="2026-11-11",
        candidates=candidates,
    )


def diagnostic_frame(value, tokenizer, binding):
    # Wider bounds describe rejected controls; they never authorize admission.
    inner = capacity.serialize_joint_cohort(
        value,
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        tokenizer_sha256=binding,
        special_ids=frozenset(tokenizer.all_special_ids),
        max_tokens=100_000,
        max_bytes=1_000_000,
    )
    return dict(
        tokens=len(inner.input_ids) + 2, payload_bytes=len(inner.payload_json.encode("utf-8"))
    )


def prepare_cases(tokenizer, binding):
    cases = []

    def add(name, value, available, reason=None):
        cases.append(
            dict(
                id=name,
                request=value,
                request_sha256=hashlib.sha256(
                    json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                supported_target_ids=[c["id"] for c in value["candidates"]],
                expected_availability="admitted" if available else "unavailable",
                expected_error_fragment=reason,
                semantic_review_status="pending",
                lineage="shared-boundary-fixtures-not-training-or-independent-semantic-evaluation",
            )
        )

    add("empty", request(0), True)
    add("one-fact", request(1), True)
    add("four-facts", request(4), True)
    add("five-facts", request(5), True)
    add("eight-owned-sources", request(1, 8), True)
    add("nine-owned-sources", request(1, 9), False, "eight-source")
    value = request(5, 8)
    source = value["candidates"][-1]["sources"][-1]
    original = source["excerpt"] + "\nRetained diagnostic filler:"
    repeats = 8192 - diagnostic_frame(value, tokenizer, binding)["tokens"]
    for _ in range(16):
        source["excerpt"] = original + " x" * repeats
        actual = diagnostic_frame(value, tokenizer, binding)["tokens"]
        if actual == 8192:
            break
        repeats += 8192 - actual
        if repeats < 0:
            raise ValueError("Full-source fixture already exceeds the token bound")
    assert diagnostic_frame(value, tokenizer, binding)["tokens"] == 8192
    add("exact-token-bound", deepcopy(value), True)
    source["excerpt"] += " x"
    assert diagnostic_frame(value, tokenizer, binding)["tokens"] == 8193
    add("token-overflow", value, False, "token budget")
    value = request(5, 8)
    missing = 64000 - diagnostic_frame(value, tokenizer, binding)["payload_bytes"]
    assert missing > 0
    value["candidates"][-1]["sources"][-1]["excerpt"] += " " * missing
    assert diagnostic_frame(value, tokenizer, binding)["payload_bytes"] == 64000
    assert diagnostic_frame(value, tokenizer, binding)["tokens"] <= 8192
    add("exact-byte-bound", deepcopy(value), True)
    value["candidates"][-1]["sources"][-1]["excerpt"] += " "
    assert diagnostic_frame(value, tokenizer, binding)["payload_bytes"] == 64001
    add("byte-overflow", value, False, "byte budget")
    add("six-facts", request(6), False, "zero through five")
    return cases


def save(path, value):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def observe_controls(cases, tokenizer, binding, output):
    outcomes, frames = [], []
    active = None
    value = None
    try:
        for case in cases:
            active = case["id"]
            value = deepcopy(case["request"])
            before = deepcopy(value)
            outcome = dict(id=active)
            try:
                framed = frame_local_support(value, tokenizer, binding)
            except ValueError as error:
                outcome.update(
                    availability="unavailable", error_type=type(error).__name__, error=str(error)
                )
            else:
                frames.append(dict(id=active, **asdict(framed)))
                outcome.update(
                    availability="admitted",
                    tokens=len(framed.input_ids),
                    payload_bytes=len(framed.payload_json.encode()),
                    frame_sha256=framed.frame_sha256,
                )
            outcomes.append(outcome)
            save(output / "outcomes.json", outcomes)
            save(output / "frames.json", frames)
            assert value == before
            assert outcome["availability"] == case["expected_availability"]
            if outcome["availability"] == "unavailable":
                assert case["expected_error_fragment"] in outcome["error"]
            else:
                assert json.loads(framed.payload_json) == value
                assert framed.target_ids == tuple(c["id"] for c in value["candidates"])
            active = None
        return outcomes, frames
    except BaseException as error:
        observed_ids = {row["id"] for row in outcomes}
        failure = dict(
            status="failed",
            failed_control=active,
            error_type=type(error).__name__,
            error=str(error),
            expected_controls=len(cases),
            expected_answerable_controls=sum(bool(c["supported_target_ids"]) for c in cases),
            expected_supported_target_opportunities=sum(
                len(c["supported_target_ids"]) for c in cases
            ),
            observed_controls=len(outcomes),
            unattempted_controls=[
                c["id"] for c in cases if c["id"] not in observed_ids and c["id"] != active
            ],
            semantic_recall=None,
            semantic_precision=None,
        )
        save(output / "outcomes.json", outcomes)
        save(output / "frames.json", frames)
        save(output / "active-input.json", value)
        save(output / "failure.json", failure)
        raise


def _run(model_root, output):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for name in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "TYPESAFE_API_KEY",
        "JEV_API_KEY",
    ):
        os.environ.pop(name, None)
    started = time.perf_counter()
    original = json.loads((model_root / "freeze.json").read_bytes())
    tokenizer, binding = capacity.tokenizer_at(model_root)
    versions = capacity.runtime_versions()
    assert binding == original["tokenizer_binding"] and versions == original["versions"]
    source_paths = [
        ROOT / "benchmarks/local_memory" / name
        for name in (
            "local_support_v2_controls.py",
            "local_support_v2_admission.py",
            "bidirectional_support_capacity.py",
            "joint_support_frame.py",
            "joint_support_capacity.py",
        )
    ]
    source_paths += [
        ROOT / "tests" / name
        for name in ("test_local_support_v2_admission.py", "test_local_support_v2_controls.py")
    ]
    bound = {str(p.relative_to(ROOT)): capacity.sha(p) for p in source_paths}
    cases = prepare_cases(tokenizer, binding)
    assert tuple(case["id"] for case in cases) == CONTROL_IDS
    save(output / "cases.json", cases)
    freeze = dict(
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        source_files=bound,
        tokenizer_binding=binding,
        versions=versions,
        cases_sha256=capacity.sha(output / "cases.json"),
        model_forwards=0,
        hosted_calls=0,
    )
    save(output / "freeze.json", freeze)
    outcomes, frames = observe_controls(cases, tokenizer, binding, output)
    accounting = account_controls(cases, outcomes)
    assert accounting["controls"] == 11 and accounting["answerable_controls"] == 10
    assert accounting["unavailable_answerable_controls"] == 4
    assert accounting["supported_target_opportunities"] == 38
    assert accounting["unavailable_supported_target_opportunities"] == 17
    assert all(capacity.sha(ROOT / name) == digest for name, digest in bound.items())
    after_files = {
        p.name: capacity.sha(p)
        for p in (model_root / "model").iterdir()
        if p.is_file() and p.suffix != ".safetensors"
    }
    assert hashlib.sha256(json.dumps(after_files, sort_keys=True).encode()).hexdigest() == binding
    assert capacity.runtime_versions() == versions
    assert capacity.sha(output / "cases.json") == freeze["cases_sha256"]
    save(output / "accounting.json", accounting)
    result = dict(
        status="tokenizer_and_admission_controls_passed_not_semantic_acceptance",
        elapsed_seconds=time.perf_counter() - started,
        controls=11,
        admitted=len(frames),
        unavailable=11 - len(frames),
        accounting=accounting,
        frozen_bindings_unchanged=True,
        dependency_versions_unchanged=True,
        source_commit=freeze["source_commit"],
        tokenizer_binding=binding,
        files={p.name: capacity.sha(p) for p in output.iterdir() if p.is_file()},
        model_forwards=0,
        hosted_calls=0,
        training_ready=False,
        limits="Synthetic availability controls only, outside training and semantic holdout counts. Not a timing, model-quality, live-runtime or pilot result.",
    )
    save(output / "result.json", result)
    return result


def run(model_root, output):
    # Refuse to overwrite an earlier attempt, including a failed preparation.
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        return _run(model_root, output)
    except BaseException as error:
        failure_path = output / "failure.json"
        if failure_path.exists():
            failure = json.loads(failure_path.read_bytes())
        else:
            outcomes_path = output / "outcomes.json"
            observed = json.loads(outcomes_path.read_bytes()) if outcomes_path.exists() else []
            observed_ids = {row["id"] for row in observed}
            failure = dict(
                expected_controls=11,
                expected_answerable_controls=10,
                expected_supported_target_opportunities=38,
                observed_controls=len(observed),
                unattempted_controls=[
                    identity for identity in CONTROL_IDS if identity not in observed_ids
                ],
                semantic_recall=None,
                semantic_precision=None,
            )
        failure.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            elapsed_seconds=time.perf_counter() - started,
            model_forwards=0,
            hosted_calls=0,
            training_ready=False,
        )
        save(output / "result.json", failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.model_root.resolve(), arguments.output.resolve()), indent=2))
