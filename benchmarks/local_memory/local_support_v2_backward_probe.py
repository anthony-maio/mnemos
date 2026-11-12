"""Prepare and run one contained synthetic full-encoder backward probe."""

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

_spec = importlib.util.spec_from_file_location(
    "local_support_v2_probe_learning", Path(__file__).with_name("local_support_v2_learning.py")
)
learning = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = learning
_spec.loader.exec_module(learning)
ROOT = learning.ROOT
MODEL_ROOT = ROOT / ".venv/bidirectional-support-capacity-v1"
MODEL_PYTHON = ROOT / ".venv/reranker-runtime-v1/Scripts/python.exe"
CONTROL_ROOT = ROOT / ".venv/local-support-v2-boundary-controls-v1"
CONTROL_BINDINGS = {
    "cases.json": "ea51fae9afa1a19f7a705daa1775d3f733966d813f976b466008b525e415c893",
    "frames.json": "eb4d757162d278ded6eac5fcf150b3c70945684ff094344c17f03e5ac90eee10",
}


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify(files):
    for path, expected in files.items():
        if learning.sha(path) != expected:
            raise ValueError("Frozen binding changed: " + str(path))


def capacity():
    return learning.load_sibling("local_support_v2_admission").capacity


def assert_interpreter():
    if Path(sys.executable).resolve() != MODEL_PYTHON.resolve():
        raise ValueError("The exact frozen model interpreter is required")


def environment():
    names = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "COMSPEC",
        "PATHEXT",
        "USERPROFILE",
        "LOCALAPPDATA",
        "APPDATA",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "OS",
        "HOMEDRIVE",
        "HOMEPATH",
    }
    result = {k: v for k, v in os.environ.items() if k.upper() in names}
    result.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONIOENCODING="utf-8",
        CUDA_VISIBLE_DEVICES="0",
    )
    return result


def prepare(destination):
    assert_interpreter()
    if not destination.resolve().is_relative_to((ROOT / ".venv").resolve()):
        raise ValueError("Probe artifacts must stay in this checkout private environment")
    destination.mkdir(parents=True, exist_ok=False)
    recipe = learning.load_recipe()
    encoder = capacity()
    original = read(MODEL_ROOT / "freeze.json")
    versions = encoder.runtime_versions()
    if versions != original["versions"] or len(versions) != 25:
        raise ValueError("Model dependency inventory changed")
    examples, labels, data_bindings = learning.load_corpus()
    rows = learning.training_rows(examples, labels)
    orders = learning.epoch_orders([r["id"] for r in rows], recipe)
    save(destination / "training-weights.json", rows)
    save(destination / "epoch-orders.json", orders)
    verify({str(CONTROL_ROOT / name): digest for name, digest in CONTROL_BINDINGS.items()})
    controls = read(CONTROL_ROOT / "cases.json")
    control_frames = {f["id"]: f for f in read(CONTROL_ROOT / "frames.json")}
    full = next(c for c in controls if c["id"] == "exact-token-bound")
    frame = {k: v for k, v in control_frames[full["id"]].items() if k != "id"}
    if (
        len(frame["input_ids"]) != 8192
        or len(frame["target_ids"]) != 5
        or any(len(c["sources"]) != 8 for c in full["request"]["candidates"])
    ):
        raise ValueError("Full probe input inventory changed")
    if json.loads(frame["payload_json"]) != full["request"]:
        raise ValueError("Full probe sources do not bind the frame")
    save(destination / "probe-request.json", full["request"])
    save(destination / "probe-frame.json", frame)
    expected_frames = {}
    for relative in recipe["corpus_receipts"]:
        audit = read(
            ROOT / "benchmarks/local_memory/data" / Path(relative).parent / "frame-audit.json"
        )
        expected_frames.update({r["id"]: r["frame_sha256"] for r in audit["rows"]})
    parity = [
        dict(id=e["id"], request=e["request"], frame_sha256=expected_frames[e["id"]])
        for e in examples
    ]
    parity.extend(
        dict(
            id="boundary:" + c["id"],
            request=c["request"],
            frame_sha256=control_frames[c["id"]]["frame_sha256"],
        )
        for c in controls
        if c["expected_availability"] == "admitted"
    )
    if len(parity) != 123 or len({r["id"] for r in parity}) != 123:
        raise ValueError("Deployment parity set changed")
    save(destination / "deployment-parity-requests.json", sorted(parity, key=lambda r: r["id"]))
    sources = [
        Path(__file__),
        Path(learning.__file__),
        Path(learning.__file__).with_name("local_support_v2_recipe.json"),
        ROOT / "tests/test_local_support_v2_learning.py",
        ROOT / "docs/plans/2026-11-12-local-support-learning-recipe-v1.md",
        ROOT / "benchmarks/native_hosts/windows_owned_job.py",
        Path(__file__).with_name("local_support_v2_owned_probe.py"),
    ]
    sources += [
        Path(__file__).with_name(name + ".py")
        for name in (
            "local_support_v2_admission",
            "bidirectional_support_capacity",
            "joint_support_frame",
            "joint_support_data",
            "joint_support_loss",
            "joint_support_capacity",
        )
    ]
    sources += [ROOT / path for path in data_bindings["files"]]
    sources += [CONTROL_ROOT / name for name in CONTROL_BINDINGS]
    sources += [p for p in (MODEL_ROOT / "model").iterdir() if p.is_file()]
    sources += [MODEL_ROOT / "freeze.json", MODEL_PYTHON]
    site = Path(sys.prefix) / "Lib/site-packages"
    sources += [
        site / path
        for path in (
            "torch/optim/adamw.py",
            "torch/optim/optimizer.py",
            "torch/amp/grad_scaler.py",
            "torch/utils/checkpoint.py",
            "torch/cuda/memory.py",
            "torch/nn/utils/clip_grad.py",
            "torch/nn/functional.py",
            "transformers/modeling_utils.py",
            "transformers/modeling_layers.py",
            "transformers/models/modernbert/modeling_modernbert.py",
            "transformers/models/modernbert/configuration_modernbert.py",
        )
    ]
    sources += [p for p in destination.iterdir() if p.is_file()]
    files = {str(p.resolve()): learning.sha(p) for p in sources}
    if learning.sha(MODEL_ROOT / "model/model.safetensors") != recipe["weight_sha256"]:
        raise ValueError("Checkpoint changed")
    freeze = dict(
        status="prepared_for_review_no_model_computation",
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        recipe=recipe,
        files=files,
        versions=versions,
        python=str(MODEL_PYTHON.resolve()),
        python_version=platform.python_version(),
        tokenizer_binding=original["tokenizer_binding"],
        model_root=str(MODEL_ROOT.resolve()),
        frame_sha256=frame["frame_sha256"],
        training_cases=56,
        training_targets=138,
        parity_frames=123,
        gpu=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
    )
    save(destination / "freeze.json", freeze)
    verify(files)
    return dict(
        status=freeze["status"],
        source_commit=freeze["source_commit"],
        bound_files=len(files),
        training_cases=56,
        training_targets=138,
        parity_frames=123,
        freeze_sha256=learning.sha(destination / "freeze.json"),
    )


def record_failure(root, result, error):
    expected = result.get("expected_microbatches", 4)
    completed = len(result.get("microbatches", []))
    active = result.get("active_microbatch")
    result.update(
        status="failed",
        error_type=type(error).__name__,
        error=str(error),
        completed_microbatches=completed,
        uncompleted_microbatches=expected - completed,
        unattempted_microbatches=[i for i in range(completed + 1, expected + 1) if i != active],
    )
    save(root / "result.json", result)


def finalize_child_record(record, path, checks_passed):
    try:
        child = read(path)
        status = child.get("status")
        if status not in {
            "synthetic_backward_probe_passed",
            "not_started_resource_unavailable",
            "failed",
        }:
            raise ValueError("Unknown child terminal status")
        record.update(child_terminal_verified=True, child_status=status)
        if status == "not_started_resource_unavailable" and checks_passed:
            record["status"] = status
        elif status != "synthetic_backward_probe_passed":
            record["status"] = "failed"
    except Exception as error:
        record.update(
            status="failed",
            child_terminal_verified=False,
            child_result_error_type=type(error).__name__,
        )
    return record


def run(root):
    assert_interpreter()
    if not root.resolve().is_relative_to((ROOT / ".venv").resolve()):
        raise ValueError("Probe execution must stay in this checkout private environment")
    freeze = read(root / "freeze.json")
    recipe = learning.load_recipe()
    if (root / "attempt.started").exists():
        raise ValueError("This numerical attempt has already started")
    if (root / "result.json").exists():
        previous = read(root / "result.json")
        if previous.get("status") != "not_started_resource_unavailable":
            raise ValueError("Refusing to overwrite an earlier terminal result")
        number = len(list(root.glob("precheck-*.json"))) + 1
        with (root / f"precheck-{number}.json").open("x", encoding="utf-8") as handle:
            json.dump(previous, handle, indent=2)
    result = dict(
        status="failed",
        stage="verify_inputs",
        expected_microbatches=4,
        microbatches=[],
        optimizer_steps_verified=0,
        corpus_training=False,
        weights_saved=False,
    )
    started = time.perf_counter()
    torch = None

    def stage(name):
        result["stage"] = name
        result["elapsed_seconds"] = time.perf_counter() - started
        if torch is not None:
            try:
                result.update(
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
            except Exception as error:
                result["memory_observation_error"] = type(error).__name__
        save(root / "progress.json", result)

    try:
        verify(freeze["files"])
        if recipe != freeze["recipe"] or capacity().runtime_versions() != freeze["versions"]:
            raise ValueError("Frozen recipe or dependency versions changed")
        import torch
        import transformers

        site = (Path(sys.prefix) / "Lib/site-packages").resolve()
        if not Path(torch.__file__).resolve().is_relative_to(site) or not Path(
            transformers.__file__
        ).resolve().is_relative_to(site):
            raise ValueError("Model imports escaped the frozen environment")
        if torch.__version__ != "2.10.0+cu128" or transformers.__version__ != "4.57.6":
            raise ValueError("Unexpected model runtime")
        torch.set_num_threads(recipe["cpu_threads"])
        torch.set_num_interop_threads(recipe["cpu_interop_threads"])
        if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4080":
            raise ValueError("Unexpected CUDA device")
        free, total = torch.cuda.mem_get_info(0)
        result.update(free_before_bytes=free, total_device_bytes=total)
        if free < recipe["backward_probe"]["minimum_free_bytes"]:
            result.update(
                status="not_started_resource_unavailable",
                stage="free_memory_precheck",
                elapsed_seconds=time.perf_counter() - started,
            )
            save(root / "result.json", result)
            return result
        with (root / "attempt.started").open("x", encoding="utf-8") as handle:
            json.dump(
                dict(
                    freeze_sha256=learning.sha(root / "freeze.json"),
                    free_bytes=free,
                    total_bytes=total,
                ),
                handle,
            )
        torch.cuda.set_per_process_memory_fraction(
            recipe["backward_probe"]["allocator_limit_bytes"] / total, 0
        )
        torch.manual_seed(recipe["seed"])
        torch.cuda.manual_seed_all(recipe["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(recipe["sdpa_backends"]["flash"])
        torch.backends.cuda.enable_mem_efficient_sdp(recipe["sdpa_backends"]["memory_efficient"])
        torch.backends.cuda.enable_math_sdp(recipe["sdpa_backends"]["math"])
        torch.cuda.reset_peak_memory_stats()
        stage("tokenizer_validation")
        tokenizer, binding = capacity().tokenizer_at(MODEL_ROOT)
        frame = learning.load_sibling("local_support_v2_admission").frame_local_support(
            read(root / "probe-request.json"), tokenizer, binding
        )
        if (
            binding != freeze["tokenizer_binding"]
            or frame.frame_sha256 != freeze["frame_sha256"]
            or json.loads(json.dumps(asdict(frame))) != read(root / "probe-frame.json")
        ):
            raise ValueError("Full probe frame changed")
        stage("load_model")
        model, loading = learning.build_model(MODEL_ROOT, recipe)
        result.update(
            backbone_parameters=sum(p.numel() for p in model.backbone.parameters()),
            total_parameters=sum(p.numel() for p in model.parameters()),
            layers=len(model.backbone.layers),
        )
        optimizer = learning.make_optimizer(model, recipe)
        names = {id(p): name for name, p in model.named_parameters()}
        groups = [
            dict(
                name=g["name"],
                lr=g["lr"],
                weight_decay=g["weight_decay"],
                parameters=[
                    dict(
                        name=names[id(p)], shape=list(p.shape), numel=p.numel(), dtype=str(p.dtype)
                    )
                    for p in g["params"]
                ],
            )
            for g in optimizer.param_groups
        ]
        save(root / "optimizer-groups.json", groups)
        save(root / "loading.json", loading)
        result["initial_head_sha256"] = learning.parameter_digest(model.classifier)
        body_before = learning.parameter_digest(model.backbone)
        head_before = result["initial_head_sha256"]
        calls = dict(backbone=0, layers=[0] * recipe["backbone_layers"])

        def count_body(_module, _inputs):
            calls["backbone"] += 1

        model.backbone.register_forward_pre_hook(count_body)
        for index, layer in enumerate(model.backbone.layers):

            def count_layer(_module, _inputs, index=index):
                calls["layers"][index] += 1

            layer.register_forward_pre_hook(count_layer)
        scaler = torch.amp.GradScaler("cuda", **recipe["scaler"])
        optimizer.zero_grad(set_to_none=True)
        for number in range(1, 5):
            result["active_microbatch"] = number
            stage("forward_" + str(number))
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(frame)
                loss = learning.weighted_micro_loss(
                    logits,
                    recipe["backward_probe"]["targets"],
                    [recipe["backward_probe"]["target_weight"]] * 5,
                    recipe,
                )
            stage("backward_" + str(number))
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            result["microbatches"].append(
                dict(
                    index=number,
                    weighted_loss=float(loss.detach()),
                    tokens=len(frame.input_ids),
                    targets=len(frame.target_ids),
                    frame_sha256=frame.frame_sha256,
                )
            )
            result["active_microbatch"] = None
            save(root / "microbatches.json", result["microbatches"])
            stage("completed_" + str(number))
        stage("unscale_and_clip")
        scaler.unscale_(optimizer)
        result["gradient_activity"] = learning.clip_gradients(model, recipe)
        stage("optimizer_step")
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() != recipe["scaler"]["init_scale"]:
            raise ValueError("AMP scale changed or optimizer step was skipped")
        learning.validate_updated_state(model, optimizer, 1)
        result["optimizer_steps_verified"] = 1
        stage("verify_update")
        result.update(
            body_changed=learning.parameter_digest(model.backbone) != body_before,
            head_changed=learning.parameter_digest(model.classifier) != head_before,
            scaler=float(scaler.get_scale()),
            forward_entries=calls,
        )
        if not result["body_changed"] or not result["head_changed"]:
            raise ValueError("Expected full gradient update or forward inventory missing")
        learning.validate_probe_inventory(calls, recipe)
        torch.cuda.synchronize()
        stage("terminal_bindings")
        verify(freeze["files"])
        if capacity().runtime_versions() != freeze["versions"]:
            raise ValueError("Dependency versions changed")
        if (
            torch.cuda.max_memory_reserved() > recipe["backward_probe"]["allocator_limit_bytes"]
            or torch.cuda.max_memory_allocated() > recipe["backward_probe"]["allocator_limit_bytes"]
        ):
            raise ValueError("Declared allocator cap exceeded")
        result.update(
            status="synthetic_backward_probe_passed",
            completed_microbatches=4,
            uncompleted_microbatches=0,
            bindings_unchanged=True,
            dependency_versions_unchanged=True,
            elapsed_seconds=time.perf_counter() - started,
            limits="Four synthetic full-input backward computations and one disposable optimizer update only; no corpus training, semantic score, saved trained weights or trained latency.",
        )
        save(root / "result.json", result)
        return result
    except BaseException as error:
        result["elapsed_seconds"] = time.perf_counter() - started
        if torch is not None:
            try:
                result.update(
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
            except Exception:
                pass
        record_failure(root, result, error)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    value = (
        prepare(args.directory.resolve())
        if args.mode == "prepare"
        else run(args.directory.resolve())
    )
    print(json.dumps(value, indent=2))
    if value.get("status") == "not_started_resource_unavailable":
        raise SystemExit(2)
