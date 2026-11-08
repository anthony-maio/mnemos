"""Fixed eight-layer capacity probe. No labels, training or semantic decisions."""

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


def request(count, sources=1):
    return dict(
        query="Synthetic computation only: inspect the retained entries.",
        reference_date="2026-11-08",
        candidates=[
            dict(
                id=f"f{i}",
                version=1,
                content=f"Entry {i} contains a sample.",
                assertion_status="asserted",
                sources=[
                    dict(
                        id=f"s{i}-{j}",
                        source_event_id=f"e{i}-{j}",
                        excerpt="Synthetic retained sample.",
                    )
                    for j in range(sources)
                ],
            )
            for i in range(count)
        ],
    )


def prepare(destination):
    tokenizer, encode, files, binding = tokenizer_setup()
    cases = [
        dict(name="empty", request=request(0)),
        dict(name="one-short", request=request(1)),
        dict(name="five-short", request=request(5)),
    ]
    for sources in (1, 8):
        item = request(5, sources)
        base = len(encode(item).input_ids)
        original = item["candidates"][-1]["sources"][-1]["excerpt"]
        repeats = TOKEN_LIMIT - base
        for _ in range(4):
            item["candidates"][-1]["sources"][-1]["excerpt"] = original + " x" * repeats
            actual = len(encode(item, limit=TOKEN_LIMIT + 100).input_ids)
            if actual == TOKEN_LIMIT:
                break
            repeats += TOKEN_LIMIT - actual
        assert len(encode(item).input_ids) == TOKEN_LIMIT
        cases.append(dict(name=f"five-token-bound-{sources}-sources", request=item))
    item = request(5)
    initial = len(encode(item).payload_json.encode("utf-8"))
    item["candidates"][-1]["sources"][-1]["excerpt"] += " " * (BYTE_LIMIT - initial)
    assert len(encode(item).payload_json.encode("utf-8")) == BYTE_LIMIT
    cases.append(dict(name="five-byte-bound", request=item))
    frames = []
    for case in cases:
        frame = encode(case["request"])
        frames.append(
            dict(
                name=case["name"],
                tokens=len(frame.input_ids),
                payload_bytes=len(frame.payload_json.encode("utf-8")),
                target_ids=frame.target_ids,
                frame_sha256=frame.frame_sha256,
            )
        )
    for item, message in (
        (json.loads(json.dumps(cases[3]["request"])), "token"),
        (json.loads(json.dumps(cases[-1]["request"])), "byte"),
    ):
        item["candidates"][-1]["sources"][-1]["excerpt"] += " x"
        try:
            encode(item)
        except ValueError as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError("Overflow was accepted")
    import torch
    import transformers

    assert torch.__version__ == "2.10.0+cu128"
    assert transformers.__version__ == "4.57.6"
    assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4080"
    for path in (
        Path(__file__),
        Path(__file__).with_name("joint_support_head.py"),
        Path(__file__).with_name("reduced_depth_support.py"),
        Path(__file__).with_name("joint_support_frame.py"),
        ROOT / "docs/plans/2026-11-08-reduced-depth-support-v1.md",
    ):
        files[str(path)] = digest(path)
    destination.mkdir(exist_ok=False)
    write(destination / "requests.json", cases)
    write(destination / "frames.json", frames)
    for name in ("requests.json", "frames.json"):
        files[str(destination / name)] = digest(destination / name)
    freeze = dict(
        files=files,
        tokenizer_binding=binding,
        token_limit=TOKEN_LIMIT,
        byte_limit=BYTE_LIMIT,
        warmups_per_shape=5,
        repetitions_per_shape=50,
        seed=2026110807,
        device=torch.cuda.get_device_name(0),
        dtype="float16",
        attention="sdpa",
        torch=torch.__version__,
        transformers=transformers.__version__,
        python=platform.python_version(),
        batch_size=1,
        original_depth=28,
        retained_depth=8,
        cpu_threads=4,
        necessary_p95_ceiling_ms=500,
        scope="computation_only_no_labels_no_semantic_grading",
        driver=subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    )
    write(destination / "freeze.json", freeze)
    print(
        json.dumps(dict(freeze_sha256=digest(destination / "freeze.json"), frames=frames)),
        flush=True,
    )


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
        prefix = load_module("reduced_depth_support")
        assert freeze["original_depth"] == prefix.ORIGINAL_DEPTH == 28
        assert freeze["retained_depth"] == prefix.RETAINED_DEPTH == 8
        backbone = prefix.retain_fixed_prefix(backbone)
        result["original_depth"] = 28
        result["retained_depth"] = len(backbone.layers)
        head = load_module("joint_support_head").JointSupportHead(
            backbone, backbone.config.hidden_size
        )
        head = head.to(device="cuda:0", dtype=torch.float16).eval()
        torch.cuda.synchronize()
        result["load_seconds"] = time.perf_counter() - begin_load
        result["loading_info"] = loading
        result["cuda_memory_before"] = list(torch.cuda.mem_get_info())
        result["device_properties"] = str(torch.cuda.get_device_properties(0))
        calls = [0]
        hook = backbone.register_forward_hook(lambda *args: calls.__setitem__(0, calls[0] + 1))
        cases = json.loads((destination / "requests.json").read_bytes())
        frames = {r["name"]: r for r in json.loads((destination / "frames.json").read_bytes())}
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
    except BaseException as exc:
        failed = record_failure(result, rows, active_sample, exc)
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
