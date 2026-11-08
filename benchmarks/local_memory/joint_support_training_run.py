"""One frozen, supervised support-training attempt; no live product integration."""

import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TIMEOUT = 1800


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def verify(files):
    for path, expected in files.items():
        if digest(path) != expected:
            raise ValueError("Frozen binding changed: " + path)


def versions():
    return {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}


def driver():
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip()


def prepare(destination):
    import torch

    preflight_path = ROOT / ".venv/joint-support-training-tokenization-v1/preflight.json"
    bundle_path = ROOT / ".venv/joint-support-training-preparation-v1/bundle.json"
    if (
        digest(preflight_path) != "803bce271e0f8478f86cf7e8b0561be1ca47c83fb207275263c2c7ea5148278c"
        or digest(bundle_path) != "45f85f64f194b0e45913392007e3a7c2c024fd40802079953ff555b318051872"
    ):
        raise ValueError("Prepared inputs changed")
    files = dict(read(preflight_path)["files"])
    verify(files)
    paths = [
        preflight_path,
        bundle_path,
        Path(__file__),
        ROOT / "docs/plans/2026-11-08-support-training-execution-v1.md",
        ROOT / ".venv/reduced-depth-capacity-v1/requests.json",
        ROOT / ".venv/reduced-depth-capacity-v1/frames.json",
    ]
    for name in (
        "joint_support_head",
        "joint_support_adapter",
        "joint_support_backward_probe",
        "joint_support_training",
        "joint_support_baseline",
        "joint_support_grading",
        "joint_support_training_bundle",
        "joint_support_training_audit",
        "joint_support_loss",
        "joint_support_data",
        "reduced_depth_support",
        "reranker_support",
    ):
        paths.append(Path(__file__).with_name(name + ".py"))
    files.update({str(p.resolve()): digest(p) for p in paths})
    if (
        torch.__version__ != "2.10.0+cu128"
        or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4080"
    ):
        raise ValueError("Unexpected training runtime")
    freeze = dict(
        files=files,
        versions=versions(),
        python=platform.python_version(),
        driver=driver(),
        device=torch.cuda.get_device_name(0),
        bundle=str(bundle_path),
        preflight=str(preflight_path),
        timeout_seconds=TIMEOUT,
        cpu_threads=4,
        interop_threads=1,
    )
    destination.mkdir(exist_ok=False)
    write(destination / "freeze.json", freeze)
    print(
        json.dumps(
            dict(status="execution_frozen", freeze_sha256=digest(destination / "freeze.json"))
        ),
        flush=True,
    )


def run(destination):
    with (destination / "started.json").open("x", encoding="utf-8") as handle:
        json.dump(dict(started_unix=time.time()), handle)
    started = time.perf_counter()
    result = dict(status="failed", corpus_training_started=False, completed_updates=0)
    torch = None

    def stage(name, **details):
        result.update(stage=name, **details)
        result["elapsed_seconds"] = time.perf_counter() - started
        if torch is not None:
            try:
                result.update(
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
            except Exception as exc:
                result["memory_observation_error"] = str(exc)
        write(destination / "progress.json", result)

    def append(name, row):
        with (destination / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    try:
        stage("verify_freeze")
        freeze = read(destination / "freeze.json")
        result["freeze_sha256"] = digest(destination / "freeze.json")
        verify(freeze["files"])
        if (
            versions() != freeze["versions"]
            or platform.python_version() != freeze["python"]
            or driver() != freeze["driver"]
        ):
            raise ValueError("Runtime changed after freeze")
        import torch
        from transformers import AutoModelForCausalLM, Qwen3Model
        from safetensors.torch import save_file

        assert torch.cuda.get_device_name(0) == freeze["device"]
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        bundle = read(freeze["bundle"])
        recipe = bundle["recipe"]
        capacity = load("joint_support_capacity")
        tokenizer, encode, _, binding = capacity.tokenizer_setup()
        assert binding == read(freeze["preflight"])["tokenizer_binding"]
        stored = read(Path(freeze["preflight"]).with_name("joint-frames.json"))
        examples = list(bundle["examples"])
        frames = {}
        for example, saved in zip(examples, stored, strict=True):
            frame = encode(example["request"])
            assert (
                saved["id"] == example["id"]
                and json.loads(json.dumps(asdict(frame))) == saved["frame"]
            )
            frames[example["id"]] = frame
        capacity_cases = read(ROOT / ".venv/reduced-depth-capacity-v1/requests.json")
        capacity_frames = read(ROOT / ".venv/reduced-depth-capacity-v1/frames.json")
        for case, saved in zip(capacity_cases, capacity_frames, strict=True):
            identity = "capacity::" + case["name"]
            frames[identity] = encode(case["request"])
            assert frames[identity].frame_sha256 == saved["frame_sha256"]
            examples.append(dict(id=identity, split="synthetic", request=case["request"]))
        baseline_helper = load("joint_support_baseline")
        decoder = load("reranker_support")
        baseline_frames = {
            q["id"]: q["witnesses"]
            for q in read(Path(freeze["preflight"]).with_name("baseline-frames.json"))
        }
        model_path = ROOT / ".venv/reranker-support-v1/model"

        stage("baseline_load")
        torch.manual_seed(2026110803)
        torch.cuda.manual_seed_all(2026110803)
        baseline = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                torch_dtype=torch.float16,
                attn_implementation="sdpa",
            )
            .to("cuda")
            .eval()
        )
        tokenizer.padding_side = "left"
        label_ids = [bundle["baseline_frame"]["label_ids"][k] for k in ("no", "yes")]
        stage("baseline_predict")
        with torch.inference_mode():
            for question in bundle["baseline"]:
                identity = question["id"]
                try:
                    witnesses = baseline_frames[identity]
                    assert [(r["fact_id"], r["evidence_id"]) for r in question["requests"]] == [
                        (r["fact_id"], r["evidence_id"]) for r in witnesses
                    ]
                    logits = []
                    if witnesses:
                        inputs = tokenizer.pad(
                            {"input_ids": [w["input_ids"] for w in witnesses]},
                            padding=True,
                            return_tensors="pt",
                        ).to("cuda")
                        logits = (
                            baseline(**inputs, use_cache=False, logits_to_keep=1)
                            .logits[:, -1, label_ids]
                            .float()
                            .cpu()
                            .tolist()
                        )
                    decision = decoder.select_witnesses(question["requests"], logits)
                    row = dict(
                        id=identity,
                        status="assessed",
                        logits=logits,
                        decision=decision,
                        selected_ids=decision["selected_ids"],
                        token_counts=[len(w["input_ids"]) for w in witnesses],
                    )
                except Exception as exc:
                    row = dict(
                        id=identity,
                        status="unavailable",
                        selected_ids=[],
                        error=f"{type(exc).__name__}: {exc}",
                    )
                append("baseline.jsonl", row)
        del baseline
        gc.collect()
        torch.cuda.empty_cache()
        verify(freeze["files"])

        stage("candidate_load")
        torch.manual_seed(recipe["seed"])
        torch.cuda.manual_seed_all(recipe["seed"])
        backbone, loading = Qwen3Model.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        assert not any(loading[k] for k in ("missing_keys", "mismatched_keys", "error_msgs"))
        assert all(k == "lm_head.weight" for k in loading["unexpected_keys"])
        backbone = load("reduced_depth_support").retain_fixed_prefix(backbone).to("cuda")
        adapters = load("joint_support_adapter")
        adapters.install_adapters(backbone)
        head = (
            load("joint_support_head")
            .JointSupportHead(backbone, backbone.config.hidden_size)
            .to("cuda")
        )
        training_helper = load("joint_support_training")
        probe = load("joint_support_backward_probe")

        def predict(name):
            head.eval()
            observations = {}
            with torch.inference_mode():
                for example in examples:
                    identity, frame = example["id"], frames[example["id"]]
                    try:
                        with torch.autocast("cuda", dtype=torch.float16):
                            output = head(
                                [frame], pad_token_id=tokenizer.pad_token_id, max_tokens=4096
                            )[0]
                        values = output.logits.float().cpu().tolist()
                        chosen = baseline_helper.select_joint(
                            example["request"], output.target_ids, values
                        )
                        row = dict(
                            id=identity,
                            status="assessed",
                            frame_sha256=frame.frame_sha256,
                            target_ids=list(output.target_ids),
                            logits=values,
                            selected_ids=chosen,
                        )
                    except Exception as exc:
                        row = dict(
                            id=identity,
                            status="unavailable",
                            selected_ids=[],
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    append(name, row)
                    observations[identity] = row
            return observations

        stage("initial_control")
        predict("initial-control.jsonl")
        verify(freeze["files"])
        frozen_before = probe.parameter_digest(head, False)
        optimizer = torch.optim.AdamW(
            adapters.optimizer_groups(head),
            betas=tuple(recipe["betas"]),
            eps=recipe["eps"],
            weight_decay=recipe["weight_decay"],
        )
        scaler = torch.amp.GradScaler("cuda", **recipe["scaler"])
        parameters = [p for p in head.parameters() if p.requires_grad]
        rng = random.Random(recipe["seed"])
        training = list(bundle["training"])
        stage("training", corpus_training_started=True)
        head.train()
        for update in range(1, 51):
            optimizer.zero_grad(set_to_none=True)
            rng.shuffle(training)
            stage("training_accumulation", active_update=update)
            loss = training_helper.accumulate_partition(
                head,
                training,
                frames,
                tokenizer.pad_token_id,
                scaler,
                lambda: torch.autocast("cuda", dtype=torch.float16),
            )
            scaler.unscale_(optimizer)
            activity = probe.gradient_activity(head)
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, recipe["clip_norm"], error_if_nonfinite=True
            )
            previous_scale = scaler.get_scale()
            stage("training_step", active_update=update)
            scaler.step(optimizer)
            scaler.update()
            assert scaler.get_scale() >= previous_scale
            assert all(
                optimizer.state[p]["step"].item() == update and torch.isfinite(p).all()
                for p in parameters
            )
            append(
                "training.jsonl",
                dict(
                    update=update,
                    weighted_loss=loss,
                    unclipped_gradient_norm=float(norm),
                    scaler_scale=scaler.get_scale(),
                    **activity,
                ),
            )
            stage("training", completed_updates=update)
            if update % 5 == 0:
                print(json.dumps(dict(update=update, weighted_loss=loss)), flush=True)
        assert probe.parameter_digest(head, False) == frozen_before
        verify(freeze["files"])
        stage("save_final_adapters")
        save_file(
            {
                n: p.detach().cpu().contiguous()
                for n, p in head.named_parameters()
                if p.requires_grad
            },
            str(destination / "final-adapters.safetensors"),
        )
        result["final_adapters_sha256"] = digest(destination / "final-adapters.safetensors")
        stage("final_unmerged")
        before = predict("final-unmerged.jsonl")
        stage("merge")
        replacements = [
            (layer.self_attn, name, getattr(layer.self_attn, name).merged_linear())
            for layer in head.backbone.layers
            for name in ("q_proj", "v_proj")
        ]
        for attention, name, merged in replacements:
            setattr(attention, name, merged)
        head.classifier = head.classifier.half().requires_grad_(False)
        del optimizer, parameters
        gc.collect()
        stage("final_merged")
        after = predict("final-merged.jsonl")
        agreement = training_helper.merge_agreement(before, after)
        write(destination / "merge-agreement.json", agreement)
        if agreement["passed"]:
            save_file(
                {n: p.detach().cpu().contiguous() for n, p in head.state_dict().items()},
                str(destination / "final-merged.safetensors"),
            )
            result["final_merged_sha256"] = digest(destination / "final-merged.safetensors")
        verify(freeze["files"])
        result.update(
            status="execution_completed" if agreement["passed"] else "merge_agreement_failed",
            frozen_parameters_unchanged=True,
            bindings_unchanged=True,
            merge_agreement_passed=agreement["passed"],
            semantic_quality="ungraded",
        )
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stage(result.get("stage", "unknown"))
        write(destination / "result.json", result)
        print(json.dumps(result), flush=True)


def supervise(destination):
    if (destination / "started.json").exists():
        raise FileExistsError("This attempt already started")
    with (destination / "supervisor.json").open("x", encoding="utf-8") as handle:
        json.dump(dict(timeout_seconds=TIMEOUT, started_unix=time.time()), handle)
    try:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "worker", str(destination)],
            check=True,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        try:
            result = read(destination / "progress.json")
        except (OSError, ValueError):
            result = {}
        result.update(
            status="failed",
            error="Outer wall-clock timeout",
            memory_scope="last worker observation only",
        )
        write(destination / "result.json", result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run", "worker"))
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    {"prepare": prepare, "run": supervise, "worker": run}[args.mode](args.destination.resolve())
