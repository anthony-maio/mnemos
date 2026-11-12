"""One synthetic capacity screen for an inactive bidirectional support design."""

from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def _helper(name):
    key = "bidirectional_capacity_" + name
    spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


request = _helper("joint_support_capacity").request
serialize_joint_cohort = _helper("joint_support_frame").serialize_joint_cohort

REVISION = "de4ab7e77845098b7fab7f6ab9d370ddff27b19c"
MODEL = "tasksource/ModernBERT-base-nli"
WEIGHT_SHA = "86c32c52ce38b8f26e028ca959b06daee3a5f3f6947c63258bc8695dde88a465"
TOKEN_LIMIT = 8192
BYTE_LIMIT = 64000
SEED = 2026111101


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def finish_result(root, result, files):
    errors = []
    for path, expected in files.items():
        try:
            if sha(path) != expected:
                errors.append(dict(path=path, error_type="sha256_mismatch"))
        except BaseException as exc:
            errors.append(dict(path=path, error_type=type(exc).__name__))
    result.update(bindings_unchanged=bool(files) and not errors, binding_errors=errors)
    if not result["bindings_unchanged"]:
        result["status"] = "failed"
    save(root / "result.json", result)


def frame_request(value, tokenizer, binding, *, max_tokens=TOKEN_LIMIT):
    if type(max_tokens) is not int or max_tokens < 3:
        raise ValueError("Invalid token budget")
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    if type(cls) is not int or type(sep) is not int or cls < 0 or sep < 0 or cls == sep:
        raise ValueError("Expected explicit native boundary tokens")
    inner = serialize_joint_cohort(
        value,
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        tokenizer_sha256=binding,
        special_ids=frozenset(tokenizer.all_special_ids),
        max_tokens=max_tokens - 2,
        max_bytes=BYTE_LIMIT,
    )
    ids = (cls, *inner.input_ids, sep)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "format": "bidirectional-support-v1",
                "inner": inner.frame_sha256,
                "input_ids": ids,
                "cls": cls,
                "sep": sep,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return replace(
        inner,
        input_ids=ids,
        marker_starts=tuple(p + 1 for p in inner.marker_starts),
        decision_positions=tuple(p + 1 for p in inner.decision_positions),
        frame_sha256=fingerprint,
    )


def tokenizer_at(root):
    from transformers import AutoTokenizer

    files = {
        p.name: sha(p)
        for p in (root / "model").iterdir()
        if p.is_file() and p.suffix != ".safetensors"
    }
    binding = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(
        root / "model", local_files_only=True, trust_remote_code=False
    )
    return tokenizer, binding


def prepare(root):
    from huggingface_hub import HfApi, hf_hub_download

    root.mkdir(exist_ok=False)
    metadata = HfApi(endpoint="https://huggingface.co", token=False).model_info(
        MODEL, revision=REVISION, files_metadata=True
    )
    assert metadata.sha == REVISION
    names = {
        "README.md",
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    chosen = [p for p in metadata.siblings if p.rfilename in names]
    assert {p.rfilename for p in chosen} == names
    assert sum(p.size for p in chosen) <= 650_000_000
    for name in sorted(names):
        hf_hub_download(
            MODEL,
            name,
            revision=REVISION,
            local_dir=root / "model",
            token=False,
            endpoint="https://huggingface.co",
        )
    assert sha(root / "model/model.safetensors") == WEIGHT_SHA
    tokenizer, binding = tokenizer_at(root)
    encode = lambda item, limit=TOKEN_LIMIT: frame_request(
        item, tokenizer, binding, max_tokens=limit
    )
    cases = [
        dict(name="empty", request=request(0)),
        dict(name="one-short", request=request(1)),
        dict(name="five-short", request=request(5)),
    ]
    for source_count in (1, 8):
        item = request(5, source_count)
        original = item["candidates"][-1]["sources"][-1]["excerpt"]
        repeats = TOKEN_LIMIT - len(encode(item).input_ids)
        for _ in range(8):
            item["candidates"][-1]["sources"][-1]["excerpt"] = original + " x" * repeats
            actual = len(encode(item, TOKEN_LIMIT + 256).input_ids)
            if actual == TOKEN_LIMIT:
                break
            repeats += TOKEN_LIMIT - actual
        assert len(encode(item).input_ids) == TOKEN_LIMIT
        cases.append(dict(name=f"five-token-bound-{source_count}-sources", request=item))
    item = request(5)
    initial = len(encode(item).payload_json.encode())
    item["candidates"][-1]["sources"][-1]["excerpt"] += " " * (BYTE_LIMIT - initial)
    assert len(encode(item).payload_json.encode()) == BYTE_LIMIT
    cases.append(dict(name="five-byte-bound", request=item))
    for index in (3, 5):
        overflow = json.loads(json.dumps(cases[index]["request"]))
        overflow["candidates"][-1]["sources"][-1]["excerpt"] += " x"
        try:
            encode(overflow)
        except ValueError:
            pass
        else:
            raise AssertionError("Overflow accepted")
    save(root / "requests.json", cases)
    save(
        root / "frames.json", [dict(name=c["name"], **asdict(encode(c["request"]))) for c in cases]
    )
    import torch
    import transformers

    assert torch.__version__ == "2.10.0+cu128" and transformers.__version__ == "4.57.6"
    assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4080"
    paths = [root / "model" / name for name in names]
    paths += [
        Path(__file__).resolve(),
        ROOT / "benchmarks/local_memory/joint_support_frame.py",
        ROOT / "benchmarks/local_memory/joint_support_capacity.py",
        ROOT / "docs/plans/2026-11-11-bidirectional-support-design.md",
        root / "requests.json",
        root / "frames.json",
    ]
    versions = json.loads(
        subprocess.check_output(
            [sys.executable, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
            text=True,
        )
    )
    save(
        root / "freeze.json",
        dict(
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            model=MODEL,
            revision=REVISION,
            tokenizer_binding=binding,
            files={str(p): sha(p) for p in paths},
            versions=versions,
            configuration=dict(
                dtype="float16",
                attention="sdpa",
                reference_compile=False,
                intraop_threads=4,
                interop_threads=1,
                seed=SEED,
                max_tokens=TOKEN_LIMIT,
                max_bytes=BYTE_LIMIT,
                samples_per_case=20,
                necessary_p95_ms=200,
                tf32=False,
            ),
            gpu=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,memory.used",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip(),
        ),
    )
    print(
        json.dumps(
            {
                "prepared": True,
                "revision": REVISION,
                "workloads": [
                    {"name": c["name"], "tokens": len(encode(c["request"]).input_ids)}
                    for c in cases
                ],
            }
        ),
        flush=True,
    )


def run(root):
    import torch
    import transformers
    from transformers import AutoModel

    freeze = json.loads((root / "freeze.json").read_bytes())
    assert all(sha(p) == h for p, h in freeze["files"].items())
    assert torch.__version__ == "2.10.0+cu128" and transformers.__version__ == "4.57.6"
    with (root / "attempt.started").open("x") as marker:
        marker.write(str(time.time()))
    result = dict(
        status="failed",
        semantic_decisions=False,
        training_performed=False,
        source_commit=freeze["source_commit"],
        model=MODEL,
        revision=REVISION,
        freeze_sha256=sha(root / "freeze.json"),
    )
    rows = []
    stage = "loading"
    active = None
    forwards = {"count": 0}
    try:
        versions = json.loads(
            subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "list",
                    "--format=json",
                    "--disable-pip-version-check",
                ],
                text=True,
            )
        )
        assert versions == freeze["versions"]
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        assert gpu.split(",")[:3] == freeze["gpu"].split(",")[:3]
        assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4080"
        result["gpu_before_loading"] = gpu
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tokenizer, binding = tokenizer_at(root)
        assert binding == freeze["tokenizer_binding"]
        started = time.perf_counter()
        model, loading = AutoModel.from_pretrained(
            root / "model",
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            reference_compile=False,
            output_loading_info=True,
        )
        assert (
            not loading["missing_keys"]
            and not loading["mismatched_keys"]
            and not loading["error_msgs"]
        )
        assert all(k.startswith(("head.", "classifier.")) for k in loading["unexpected_keys"])
        model = model.to("cuda:0").eval()

        def count_forward(_module, _inputs):
            forwards["count"] += 1

        model.register_forward_pre_hook(count_forward)
        head = (
            torch.nn.Linear(model.config.hidden_size, 1)
            .to(device="cuda:0", dtype=torch.float16)
            .eval()
        )
        assert model.config.reference_compile is False
        result.update(
            loading=loading,
            layers=model.config.num_hidden_layers,
            backbone_parameters=sum(p.numel() for p in model.parameters()),
            load_ms=(time.perf_counter() - started) * 1000,
        )
        cases = json.loads((root / "requests.json").read_bytes())
        prepared = json.loads((root / "frames.json").read_bytes())

        def infer(item, expected=None):
            framed = frame_request(item, tokenizer, binding)
            if expected is not None:
                assert framed.frame_sha256 == expected["frame_sha256"]
            values = []
            if framed.target_ids:
                ids = torch.tensor([framed.input_ids], device="cuda:0", dtype=torch.long)
                with torch.inference_mode():
                    output = model(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        output_hidden_states=False,
                        output_attentions=False,
                        return_dict=True,
                    )
                    assert output.last_hidden_state.shape == (
                        1,
                        len(framed.input_ids),
                        model.config.hidden_size,
                    )
                    positions = torch.tensor(framed.decision_positions, device="cuda:0")
                    values = (
                        head(output.last_hidden_state[0, positions])
                        .squeeze(-1)
                        .float()
                        .cpu()
                        .tolist()
                    )
            torch.cuda.synchronize()
            assert len(values) == len(framed.target_ids) and all(math.isfinite(v) for v in values)
            return framed, values

        stage = "warmup"
        warmup = request(1)
        warmup["query"] = "Unrelated synthetic capacity warmup."
        warmup["candidates"][0]["sources"][0]["excerpt"] = "Ada owns a demonstration service."
        infer(warmup)
        for case in cases:
            if case["request"]["candidates"]:
                infer(case["request"])
        stage = "measurement"
        torch.cuda.reset_peak_memory_stats()
        for sample in range(20):
            for case, expected in zip(cases, prepared, strict=True):
                torch.cuda.synchronize()
                active = dict(name=case["name"], sample=sample)
                started = time.perf_counter()
                framed, values = infer(case["request"], expected)
                elapsed = (time.perf_counter() - started) * 1000
                row = dict(
                    status="completed",
                    name=case["name"],
                    sample=sample,
                    milliseconds=elapsed,
                    tokens=len(framed.input_ids),
                    frame_sha256=framed.frame_sha256,
                    target_ids=framed.target_ids,
                    random_head_outputs=values,
                )
                with (root / "samples.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                rows.append(row)
                active = None
        summaries = {}
        for case in cases:
            durations = sorted(r["milliseconds"] for r in rows if r["name"] == case["name"])
            assert len(durations) == 20
            summaries[case["name"]] = dict(
                p95_ms=durations[18], median_ms=(durations[9] + durations[10]) / 2, samples=20
            )
        assert len(rows) == 120 and forwards["count"] == 106
        result.update(
            status="passed" if all(r["p95_ms"] <= 200 for r in summaries.values()) else "failed",
            workloads=summaries,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
    except BaseException as exc:
        if active is not None:
            failure = dict(
                **active,
                status="failed",
                error_type=type(exc).__name__,
                elapsed_until_exception_ms=(time.perf_counter() - started) * 1000,
            )
            with (root / "samples.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(failure) + "\n")
        result.update(
            status="failed",
            stage=stage,
            error_type=type(exc).__name__,
            error=str(exc),
            completed_samples=len(rows),
        )
    finally:
        result["model_forward_calls"] = forwards["count"]
        result["expected_model_forward_calls"] = 106
        result["measured_samples_completed"] = len(rows)
        result["expected_measured_samples"] = 120
        finish_result(root, result, freeze["files"])
        print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    destination = Path(sys.argv[2]).resolve()
    if sys.argv[1] == "prepare":
        prepare(destination)
    elif sys.argv[1] == "run":
        sys.exit(run(destination))
