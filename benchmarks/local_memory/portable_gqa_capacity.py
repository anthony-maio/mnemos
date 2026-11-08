"""Frozen retained-model parity and capacity for the portable GQA candidate."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TOKEN_LIMIT = 4096
BYTE_LIMIT = 64000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def record_failure(result, rows, active_sample, exc):
    result["status"] = "failed"
    result["error"] = f"{type(exc).__name__}: {exc}"
    if active_sample is not None:
        row = {k: v for k, v in active_sample.items() if k != "start_perf"}
        row.update(
            status="failed",
            error=result["error"],
            elapsed_until_exception_ms=(time.perf_counter() - active_sample["start_perf"]) * 1000,
        )
        rows.append(row)
        return row
    return None


def tokenizer_setup():
    from transformers import AutoTokenizer

    prior = json.loads((ROOT / ".venv/reranker-support-v1/execution-freeze.json").read_bytes())
    files = {p: h for p, h in prior["files"].items() if "/model/" in p.replace("\\", "/")}
    assert len(files) == 12
    for path, expected in files.items():
        assert digest(path) == expected, path
    metadata = {p: h for p, h in files.items() if not p.endswith(".safetensors")}
    binding = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / ".venv/reranker-support-v1/model", local_files_only=True, trust_remote_code=False
    )
    serializer = load_module("joint_support_frame")

    def encode(request, limit=TOKEN_LIMIT, byte_limit=BYTE_LIMIT):
        return serializer.serialize_joint_cohort(
            request,
            lambda text: tokenizer.encode(text, add_special_tokens=False),
            tokenizer_sha256=binding,
            special_ids=frozenset(tokenizer.all_special_ids),
            max_tokens=limit,
            max_bytes=byte_limit,
        )

    return tokenizer, encode, files, binding


def prepare(destination):
    import shutil
    import torch
    import transformers
    import transformers.integrations.sdpa_attention as sdpa
    import transformers.masking_utils as masking

    prior_dir = ROOT / ".venv/joint-support-capacity-v2"
    prior_path = prior_dir / "freeze.json"
    assert digest(prior_path) == "1cc656ab8ce251cf5ebed974610080a0215cbece48d69438080446bcc07ba18b"
    freeze = json.loads(prior_path.read_bytes())
    for path, expected in freeze["files"].items():
        assert digest(path) == expected, path
    destination.mkdir(exist_ok=False)
    for name in ("requests.json", "frames.json"):
        shutil.copyfile(prior_dir / name, destination / name)
        freeze["files"][str(destination / name)] = digest(destination / name)
    for path in (
        prior_path,
        Path(__file__),
        Path(__file__).with_name("portable_grouped_sdpa.py"),
        ROOT / "docs/plans/2026-11-08-portable-gqa-capacity-v1.md",
        Path(sdpa.__file__),
        Path(masking.__file__),
        Path(torch.nn.functional.__file__),
    ):
        freeze["files"][str(path)] = digest(path)
    freeze.update(
        attention="mnemos_portable_grouped_sdpa_v1",
        parity_atol=0.02,
        parity_rtol=0.02,
        original_freeze_sha256=digest(prior_path),
    )
    write(destination / "freeze.json", freeze)
    print(json.dumps(dict(freeze_sha256=digest(destination / "freeze.json"))), flush=True)


def retained_parity(head, backbone, tokenizer, encode, cases, frames, freeze, destination, result):
    import torch

    adapter = load_module("portable_grouped_sdpa")
    name = adapter.register_prefill_adapter()
    assert name == freeze["attention"]
    records = []
    result["parity"] = records
    hidden = []
    observations = []

    def capture(module, args, output):
        hidden.append(output.last_hidden_state.detach().float().cpu())

    def observe(representation, q, k, v, mask, causal, dropout, scale, native):
        def tensor(t):
            return (
                None
                if t is None
                else dict(
                    shape=list(t.shape),
                    stride=list(t.stride()),
                    dtype=str(t.dtype),
                    device=str(t.device),
                )
            )

        observations.append(
            dict(
                representation=representation,
                query=tensor(q),
                key=tensor(k),
                value=tensor(v),
                mask=tensor(mask),
                causal=causal,
                dropout=dropout,
                scale=scale,
                enable_gqa=native,
                grad_enabled=torch.is_grad_enabled(),
                dispatch=int(
                    torch._fused_sdp_choice(
                        q,
                        k,
                        v,
                        attn_mask=mask,
                        dropout_p=dropout,
                        is_causal=causal,
                        scale=scale,
                        enable_gqa=native,
                    )
                ),
            )
        )

    hook = backbone.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for case in cases:
                row = dict(shape=case["name"], status="started", comparisons={})
                records.append(row)
                frame = encode(case["request"])
                assert frame.frame_sha256 == frames[case["name"]]["frame_sha256"]
                hidden.clear()
                backbone.config._attn_implementation = "sdpa"
                expected = head(
                    [frame], pad_token_id=tokenizer.pad_token_id, max_tokens=TOKEN_LIMIT
                )[0]
                expected_hidden = hidden.pop() if frame.target_ids else None
                assert not hidden
                observations.clear()
                for layer in backbone.layers:
                    layer.self_attn._mnemos_attention_observer = observe
                backbone.config._attn_implementation = name
                actual = head([frame], pad_token_id=tokenizer.pad_token_id, max_tokens=TOKEN_LIMIT)[
                    0
                ]
                actual_hidden = hidden.pop() if frame.target_ids else None
                assert not hidden
                assert actual.target_ids == expected.target_ids == frame.target_ids
                assert actual.frame_sha256 == expected.frame_sha256 == frame.frame_sha256
                row["dispatch"] = list(observations)
                assert len(observations) == (28 if frame.target_ids else 0)
                pairs = [
                    (
                        "logits",
                        actual.logits.detach().float().cpu(),
                        expected.logits.detach().float().cpu(),
                    )
                ]
                if frame.target_ids:
                    pairs.append(("all_final_hidden", actual_hidden, expected_hidden))
                for label, got, want in pairs:
                    assert got.shape == want.shape, label
                    finite = bool(torch.isfinite(got).all() and torch.isfinite(want).all())
                    difference = (got - want).abs()
                    failures = difference > (
                        freeze["parity_atol"] + freeze["parity_rtol"] * want.abs()
                    )
                    row["comparisons"][label] = dict(
                        finite=finite,
                        elements=got.numel(),
                        max_absolute=float(difference.max()) if difference.numel() else 0,
                        outside_tolerance=int(failures.sum()),
                        fraction_outside_tolerance=(
                            float(failures.float().mean()) if failures.numel() else 0
                        ),
                    )
                write(destination / "parity.json", records)
                assert all(
                    c["finite"] and c["outside_tolerance"] == 0 for c in row["comparisons"].values()
                ), row["comparisons"]
                row["status"] = "passed"
                write(destination / "parity.json", records)
                print(json.dumps(dict(parity_shape=case["name"], status="passed")), flush=True)
    finally:
        hook.remove()
        for layer in backbone.layers:
            if hasattr(layer.self_attn, "_mnemos_attention_observer"):
                del layer.self_attn._mnemos_attention_observer
        write(destination / "parity.json", records)


def run(destination):
    # Exclusive marker prevents accidental restarts of an observed or failed run.
    with (destination / "started.json").open("x", encoding="utf-8") as handle:
        json.dump(dict(started_unix=time.time()), handle)
    rows = []
    active_sample = None
    result = dict(status="failed", scope="computation_only_random_head", samples=[])
    try:
        freeze = json.loads((destination / "freeze.json").read_bytes())
        result["freeze_sha256"] = digest(destination / "freeze.json")
        for path, expected in freeze["files"].items():
            assert digest(path) == expected, path
        import torch
        import transformers
        from transformers import Qwen3Model

        assert torch.__version__ == freeze["torch"]
        assert transformers.__version__ == freeze["transformers"]
        assert platform.python_version() == freeze["python"]
        assert torch.cuda.get_device_name(0) == freeze["device"]
        assert (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            == freeze["driver"]
        )
        torch.manual_seed(freeze["seed"])
        torch.cuda.manual_seed_all(freeze["seed"])
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        result["backend_flags"] = dict(
            flash=torch.backends.cuda.flash_sdp_enabled(),
            efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
            math=torch.backends.cuda.math_sdp_enabled(),
            cudnn=torch.backends.cuda.cudnn_sdp_enabled(),
        )
        assert all(result["backend_flags"].values())
        begin_load = time.perf_counter()
        tokenizer, encode, _, binding = tokenizer_setup()
        assert binding == freeze["tokenizer_binding"]
        backbone, loading = Qwen3Model.from_pretrained(
            ROOT / ".venv/reranker-support-v1/model",
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        assert (
            not loading["missing_keys"]
            and not loading["mismatched_keys"]
            and not loading["error_msgs"]
        ), loading
        assert all(k == "lm_head.weight" for k in loading["unexpected_keys"]), loading
        head = load_module("joint_support_head").JointSupportHead(
            backbone, backbone.config.hidden_size
        )
        head = head.to(device="cuda:0", dtype=torch.float16).eval()
        torch.cuda.synchronize()
        result["load_seconds"] = time.perf_counter() - begin_load
        result["loading_info"] = loading
        result["cuda_memory_before"] = list(torch.cuda.mem_get_info())
        result["device_properties"] = str(torch.cuda.get_device_properties(0))
        cases = json.loads((destination / "requests.json").read_bytes())
        frames = {r["name"]: r for r in json.loads((destination / "frames.json").read_bytes())}
        retained_parity(
            head, backbone, tokenizer, encode, cases, frames, freeze, destination, result
        )
        calls = [0]
        hook = backbone.register_forward_hook(lambda *args: calls.__setitem__(0, calls[0] + 1))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with (
            torch.inference_mode(),
            (destination / "samples.jsonl").open("x", encoding="utf-8") as log,
        ):
            for phase, repeats in (("warmup", 5), ("measured", 50)):
                for repeat in range(repeats):
                    for case in cases:
                        torch.cuda.synchronize()
                        before_calls = calls[0]
                        start = time.perf_counter()
                        active_sample = dict(
                            phase=phase, repeat=repeat, shape=case["name"], start_perf=start
                        )
                        frame = encode(case["request"])
                        assert frame.frame_sha256 == frames[case["name"]]["frame_sha256"]
                        outputs = head(
                            [frame], pad_token_id=tokenizer.pad_token_id, max_tokens=TOKEN_LIMIT
                        )
                        values = outputs[0].logits.detach().float().cpu().tolist()
                        assert outputs[0].target_ids == frame.target_ids
                        assert len(values) == len(frame.target_ids) and all(
                            math.isfinite(v) for v in values
                        )
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - start) * 1000
                        forwards = calls[0] - before_calls
                        assert forwards == int(bool(frame.target_ids))
                        row = dict(
                            phase=phase,
                            repeat=repeat,
                            shape=case["name"],
                            elapsed_ms=elapsed,
                            forward_calls=forwards,
                            targets=len(values),
                        )
                        rows.append(row)
                        log.write(json.dumps(row) + "\n")
                        log.flush()
                        active_sample = None
                    if repeat % 10 == 0:
                        print(
                            json.dumps(dict(phase=phase, completed_repetitions=repeat + 1)),
                            flush=True,
                        )
        hook.remove()
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        summaries = {}
        for case in cases:
            times = sorted(
                r["elapsed_ms"]
                for r in rows
                if r["phase"] == "measured" and r["shape"] == case["name"]
            )
            assert len(times) == 50
            summaries[case["name"]] = dict(
                n=len(times),
                median_ms=(times[24] + times[25]) / 2,
                p95_ms=times[math.ceil(0.95 * len(times)) - 1],
                max_ms=times[-1],
            )
        result["shapes"] = summaries
        for path, expected in freeze["files"].items():
            assert digest(path) == expected, path
        result["bindings_unchanged"] = True
        result["status"] = (
            "necessary_capacity_screen_passed"
            if all(s["p95_ms"] < 500 for s in summaries.values())
            else "capacity_screen_failed"
        )
        del head, backbone
        torch.cuda.empty_cache()
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        failed = record_failure(result, rows, active_sample, exc)
        if result.get("parity") and result["parity"][-1]["status"] == "started":
            result["parity"][-1].update(status="failed", error=f"{type(exc).__name__}: {exc}")
            write(destination / "parity.json", result["parity"])
        if failed is not None:
            try:
                with (destination / "samples.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(failed) + "\n")
            except OSError as log_exc:
                result["failure_log_error"] = str(log_exc)
        raise
    finally:
        result["samples"] = rows
        write(destination / "result.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    destination = arguments.destination.resolve()
    (prepare if arguments.mode == "prepare" else run)(destination)
