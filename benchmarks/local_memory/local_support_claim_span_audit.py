"""Offline tokenizer and readout-binding audit; never constructs an encoder."""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

name = "claim_span_audit_contract"
spec = importlib.util.spec_from_file_location(
    name, Path(__file__).with_name("local_support_claim_spans.py")
)
spans = importlib.util.module_from_spec(spec)
sys.modules[name] = spans
spec.loader.exec_module(spans)
learning = spans.learning
ROOT = learning.ROOT


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def audit(root):
    probe = learning.load_sibling("local_support_v2_backward_probe")
    probe.assert_interpreter()
    root = root.resolve()
    if not root.is_relative_to((ROOT / ".venv").resolve()):
        raise ValueError("Private audit root required")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("Freeze the clean source before tokenizer audit")
    root.mkdir(parents=True, exist_ok=False)
    previous = ROOT / ".venv/local-support-v2-corpus-run-v1-repair1"
    original = read(previous / "freeze.json")
    probe.verify(original["files"])
    archive = ROOT / ".venv/local-support-v2-corpus-run-v1-repair1.zip"
    span_failure_archive = ROOT / ".venv/local-support-claim-spans-v1-failed.zip"
    if (
        learning.sha(span_failure_archive)
        != "ebf6485c42f58d6bfd791fd954ebb5ede18a5a5ff5ddc2306c76214966f990b3"
    ):
        raise ValueError("Original Unicode mapping failure archive changed")
    if learning.sha(archive) != "568cf0fd3f208ede61991a0dcbbe5759dd247f8df5d1ca00edc98f28cfbd105f":
        raise ValueError("Original failed semantic evidence changed")
    if probe.capacity().runtime_versions() != original["versions"]:
        raise ValueError("Pinned tokenizer environment changed")
    tokenizer, binding = probe.capacity().tokenizer_at(probe.MODEL_ROOT)
    if binding != original["tokenizer_binding"]:
        raise ValueError("Pinned tokenizer binding changed")
    paths = [
        Path(__file__),
        Path(spans.__file__),
        ROOT / "tests/test_local_support_claim_spans.py",
        ROOT / "docs/plans/2026-11-12-claim-span-readout-design.md",
        previous / "freeze.json",
        previous / "frames.json",
        previous / "initial.json",
        previous / "deployment.json",
        previous / "loading.json",
        archive,
        span_failure_archive,
        ROOT / ".venv/local-support-claim-spans-v1/freeze.json",
        ROOT / ".venv/local-support-claim-spans-v1/failure.json",
    ]
    files = {**original["files"], **{str(p.resolve()): learning.sha(p) for p in paths}}
    freeze = dict(
        schema="claim-span-tokenizer-audit-v1",
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        files=files,
        versions=original["versions"],
        tokenizer_binding=binding,
    )
    save(root / "freeze.json", freeze)
    examples, labels, _ = learning.load_corpus()
    old_frames = read(previous / "frames.json")
    outputs = {arm: read(previous / (arm + ".json")) for arm in ("initial", "deployment")}
    frames, rows = {}, []
    old_readout_tokens = Counter()
    positional = {
        split: {arm: 0 for arm in outputs} for split in ("train", "tuning", "development")
    }
    for example in examples:
        identity = example["id"]
        value = example["request"]
        old = old_frames[identity]
        if len(old["marker_starts"]) != len(old["target_ids"]) or len(
            old["decision_positions"]
        ) != len(old["target_ids"]):
            raise ValueError("Original marker cardinality mismatch")
        if (
            old["target_ids"] != [c["id"] for c in value["candidates"]]
            or json.loads(old["payload_json"]) != value
        ):
            raise ValueError("Original index or payload mismatch")
        for i, (start, end) in enumerate(zip(old["marker_starts"], old["decision_positions"])):
            if (
                tokenizer.decode(old["input_ids"][start : end + 1])
                != f"\nDecision target index: {i}\n"
            ):
                raise ValueError("Original native marker or offset mismatch")
            old_readout_tokens[old["input_ids"][end]] += 1
        frame = spans.frame_claim_spans(value, tokenizer, binding)
        if list(frame.input_ids[1:-1]) != tokenizer.encode(frame.text, add_special_tokens=False):
            raise ValueError("Explicit normalization changed native token IDs")
        if (
            frame.payload_json != old["payload_json"]
            or [c.id for c in frame.claims] != old["target_ids"]
        ):
            raise ValueError("Complete payload or target identity changed")
        frames[identity] = asdict(frame)
        rows.append(
            dict(
                id=identity,
                split=example["split"],
                tokens=len(frame.input_ids),
                claims=len(frame.claims),
                claim_tokens=[len(c.tokens) for c in frame.claims],
                frame_sha256=frame.frame_sha256,
                payload_bytes=len(frame.payload_json.encode("utf-8")),
            )
        )
        for arm, observations in outputs.items():
            positional[example["split"]][arm] += observations[identity]["selected_ids"] == [
                value["candidates"][-1]["id"]
            ]
    controls = read(probe.CONTROL_ROOT / "cases.json")
    control_rows = []
    for control in controls:
        value = control["request"]
        try:
            frame = spans.frame_claim_spans(value, tokenizer, binding)
            frames["boundary:" + control["id"]] = asdict(frame)
            row = dict(
                id=control["id"],
                status="admitted",
                tokens=len(frame.input_ids),
                payload_bytes=len(frame.payload_json.encode("utf-8")),
                frame_sha256=frame.frame_sha256,
            )
        except ValueError as error:
            row = dict(id=control["id"], status="unavailable", error=str(error))
        if control["id"] not in {"exact-token-bound", "token-overflow"} and (
            row["status"] == "admitted"
        ) != (control["expected_availability"] == "admitted"):
            raise ValueError("Non-token admission changed")
        row["historical_availability"] = control["expected_availability"]
        control_rows.append(row)
    # The new format removes old markers; derive exact new token boundaries
    # without changing any original control, source text or semantic label.
    base = next(c["request"] for c in controls if c["id"] == "exact-token-bound")

    def padded(count):
        value = deepcopy(base)
        value["candidates"][-1]["sources"][-1]["excerpt"] += " x" * count
        return value

    def token_count(value):
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return len(tokenizer.encode(spans.PREAMBLE + payload, add_special_tokens=False)) + 2

    lo, hi = 0, 8192
    while lo < hi:
        mid = (lo + hi) // 2
        if token_count(padded(mid)) < 8192:
            lo = mid + 1
        else:
            hi = mid
    exact = padded(lo)
    if token_count(exact) != 8192:
        raise ValueError("Could not construct exact native token boundary")
    exact_frame = spans.frame_claim_spans(exact, tokenizer, binding)
    overflow = padded(lo + 1)
    if token_count(overflow) != 8193:
        raise ValueError("One-token overflow changed")
    try:
        spans.frame_claim_spans(overflow, tokenizer, binding)
    except ValueError as error:
        if "token budget" not in str(error):
            raise
    else:
        raise ValueError("One-token overflow incorrectly admitted")
    frames["new-boundary:exact-token"] = asdict(exact_frame)
    save(
        root / "new-token-boundary-requests.json",
        dict(
            exact=exact,
            overflow=overflow,
            semantics="Repeated derivative controls only; no new labels or independent data",
        ),
    )
    # Exercise actual native offsets on Unicode, escapes, repeated values and
    # boundary-crossing punctuation, not just the deterministic byte test codec.
    extra = deepcopy(examples[0]["request"])
    for i, c in enumerate(extra["candidates"]):
        c["content"] = 'Caf\u00e9 "40"\\none\nnot 90; \u03b1\u0301 \U0001f642'
        for s in c["sources"]:
            s["excerpt"] = 'Quoted "content" text and ' + c["content"]
    extra_frame = spans.frame_claim_spans(extra, tokenizer, binding)
    if list(extra_frame.input_ids[1:-1]) != tokenizer.encode(
        extra_frame.text, add_special_tokens=False
    ):
        raise ValueError("Unicode normalization changed native token IDs")
    frames["offset-fixture:unicode-repetition"] = asdict(extra_frame)
    save(root / "frames.json", frames)
    probe.verify(files)
    if probe.capacity().runtime_versions() != original["versions"]:
        raise ValueError("Tokenizer runtime changed")
    result = dict(
        status="model_free_span_contract_audit_passed",
        source_commit=freeze["source_commit"],
        bound_files=len(files),
        tokenizer_binding=binding,
        corpus_cases=len(rows),
        corpus_targets=sum(r["claims"] for r in rows),
        payloads_byte_identical=True,
        rows=rows,
        old_readout_token_ids=dict(old_readout_tokens),
        old_readout_token_text={str(i): tokenizer.decode([i]) for i in old_readout_tokens},
        original_tail_marker_bindings_valid=True,
        original_positional_selection_counts=positional,
        original_unused_checkpoint_tensors=read(previous / "loading.json")["unexpected_keys"],
        prior_unicode_mapping_failure_archive_sha256=learning.sha(span_failure_archive),
        historical_controls_reframed=control_rows,
        exact_new_token_bound=8192,
        new_token_overflow=8193,
        new_exact_token_frame_sha256=exact_frame.frame_sha256,
        native_unicode_fixture_sha256=extra_frame.frame_sha256,
        raw_payloads_preserved=True,
        explicit_nfc_preserves_native_token_ids=True,
        saved_frames=len(frames),
        bindings_unchanged=True,
        versions_unchanged=True,
        encoder_forwards=0,
        training_updates=0,
        hosted_calls=0,
        quality_established=False,
        latency_established=False,
        live_activation=False,
        limitation="Direct claim binding does not establish the cause of the previous failure, semantic support, source noninterference or permutation-invariant contextual states.",
    )
    save(root / "receipt.json", result)
    return {
        k: v
        for k, v in result.items()
        if k not in {"rows", "historical_controls_reframed", "original_unused_checkpoint_tensors"}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2), flush=True)
