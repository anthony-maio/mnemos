"""One disposable synthetic retained-weight optimizer step, never corpus training."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

TIMEOUT_SECONDS = 180


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_capacity():
    spec = importlib.util.spec_from_file_location(
        "capacity", Path(__file__).with_name("joint_support_capacity.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parameter_digest(head, trainable):
    digest = hashlib.sha256()
    for name, parameter in head.named_parameters():
        if parameter.requires_grad == trainable:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_activity(head):
    import torch

    parameters = [p for p in head.parameters() if p.requires_grad]
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
        raise ValueError("Missing or nonfinite trainable gradient")
    b_gradient = sum(
        float(p.grad.abs().sum())
        for name, p in head.named_parameters()
        if name.endswith(".b") and p.requires_grad
    )
    classifier_gradient = sum(float(p.grad.abs().sum()) for p in head.classifier.parameters())
    if not (b_gradient > 0 and classifier_gradient > 0):
        raise ValueError("Nonzero adapter-B and classifier gradients are required")
    return dict(adapter_b_gradient_l1=b_gradient, classifier_gradient_l1=classifier_gradient)


def run(destination):
    destination.mkdir(exist_ok=False)
    result = dict(status="failed", scope="one_disposable_synthetic_step_no_corpus_training")
    started = time.perf_counter()
    torch = None

    def stage(name):
        result["stage"] = name
        if torch is not None:
            try:
                result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            except Exception as memory_error:
                result["memory_observation_error"] = str(memory_error)
        write(destination / "progress.json", result)

    try:
        stage("imports")
        capacity = load_capacity()
        import torch
        import transformers
        from transformers import Qwen3Model

        assert torch.__version__ == "2.10.0+cu128"
        assert transformers.__version__ == "4.57.6"
        assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4080"
        torch.manual_seed(2026110808)
        torch.cuda.manual_seed_all(2026110808)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        stage("preparation")
        tokenizer, encode, files, binding = capacity.tokenizer_setup()
        request = capacity.request(5)
        original = request["candidates"][-1]["sources"][-1]["excerpt"]
        repeats = 552 - len(encode(request).input_ids)
        for _ in range(4):
            request["candidates"][-1]["sources"][-1]["excerpt"] = original + " x" * repeats
            actual = len(encode(request).input_ids)
            if actual == 552:
                break
            repeats += 552 - actual
        frame = encode(request)
        assert len(frame.input_ids) == 552
        capacity.write(destination / "request.json", request)
        for path in (
            Path(__file__),
            Path(__file__).with_name("joint_support_capacity.py"),
            Path(__file__).with_name("joint_support_head.py"),
            Path(__file__).with_name("joint_support_frame.py"),
            Path(__file__).with_name("joint_support_adapter.py"),
            Path(__file__).with_name("reduced_depth_support.py"),
            capacity.ROOT / "docs/plans/2026-11-08-support-backward-probe-v1.md",
            destination / "request.json",
        ):
            files[str(path)] = capacity.digest(path)
        freeze = dict(
            files=files,
            seed=2026110808,
            tokens=552,
            targets=5,
            tokenizer_binding=binding,
            frame_sha256=frame.frame_sha256,
            torch=torch.__version__,
            transformers=transformers.__version__,
            device=torch.cuda.get_device_name(0),
            dtype="float16",
            attention="sdpa",
            scaler=dict(init_scale=128, growth_factor=2, backoff_factor=0.5, growth_interval=2000),
        )
        capacity.write(destination / "freeze.json", freeze)
        result["freeze_sha256"] = capacity.digest(destination / "freeze.json")
        torch.cuda.reset_peak_memory_stats()
        stage("load_checkpoint")
        backbone, loading = Qwen3Model.from_pretrained(
            capacity.ROOT / ".venv/reranker-support-v1/model",
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        assert not any(loading[k] for k in ("missing_keys", "mismatched_keys", "error_msgs"))
        assert all(k == "lm_head.weight" for k in loading["unexpected_keys"])
        backbone = (
            capacity.load_module("reduced_depth_support").retain_fixed_prefix(backbone).to("cuda")
        )
        adapters = capacity.load_module("joint_support_adapter")
        adapters.install_adapters(backbone)
        head = (
            capacity.load_module("joint_support_head")
            .JointSupportHead(backbone, backbone.config.hidden_size)
            .to("cuda")
        )
        head.train()
        stage("initial_parameter_audit")
        frozen_before = parameter_digest(head, False)
        trainable_before = parameter_digest(head, True)
        optimizer = torch.optim.AdamW(
            adapters.optimizer_groups(head), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )
        scaler = torch.amp.GradScaler("cuda", **freeze["scaler"])
        stage("forward")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = head([frame], pad_token_id=tokenizer.pad_token_id, max_tokens=4096)[0].logits
            targets = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0], device="cuda")
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), targets)
        assert torch.isfinite(loss)
        stage("backward")
        scaler.scale(loss).backward()
        stage("gradient_validation")
        scaler.unscale_(optimizer)
        parameters = [p for p in head.parameters() if p.requires_grad]
        result.update(gradient_activity(head))
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        stage("optimizer_step")
        scaler.step(optimizer)
        scaler.update()
        assert scaler.get_scale() == 128
        assert all(optimizer.state[p]["step"].item() == 1 for p in parameters)
        stage("final_parameter_audit")
        assert frozen_before == parameter_digest(head, False)
        assert trainable_before != parameter_digest(head, True)
        assert all(p.grad is None for p in head.parameters() if not p.requires_grad)
        assert all(torch.isfinite(p).all() for p in parameters)
        torch.cuda.synchronize()
        for path, expected in files.items():
            assert capacity.digest(path) == expected
        result.update(
            status="synthetic_backward_passed",
            loss=float(loss.detach()),
            unclipped_gradient_norm=float(norm),
            trainable_parameters=sum(p.numel() for p in parameters),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            frozen_parameters_unchanged=True,
            bindings_unchanged=True,
            optimizer_updates=1,
            scaler_scale=scaler.get_scale(),
        )
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        stage(result.get("stage", "unknown"))
        write(destination / "result.json", result)
        print(json.dumps(result), flush=True)


def supervise(destination):
    if destination.exists():
        raise FileExistsError(destination)
    # An exclusive sidecar prevents two supervisors racing before worker mkdir.
    with destination.with_suffix(".supervisor.json").open("x", encoding="utf-8") as handle:
        json.dump(dict(timeout_seconds=TIMEOUT_SECONDS, started_unix=time.time()), handle)
    try:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(destination)],
            check=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        # subprocess.run kills and waits for its owned worker before raising.
        destination.mkdir(exist_ok=True)
        progress = destination / "progress.json"
        try:
            result = json.loads(progress.read_bytes()) if progress.exists() else {}
        except (ValueError, OSError):
            result = {}
        result.update(
            status="failed",
            error="Outer wall-clock timeout",
            timeout_seconds=TIMEOUT_SECONDS,
            memory_scope="last worker observation; unavailable during stalled call",
        )
        write(destination / "result.json", result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    (run if args.worker else supervise)(args.destination.resolve())
