"""One frozen local full-encoder learning attempt, comparison and FP16 export."""

import argparse
import copy
from dataclasses import asdict
import gc
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

spec = importlib.util.spec_from_file_location(
    "local_support_v2_training_probe",
    Path(__file__).with_name("local_support_v2_backward_probe.py"),
)
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)
learning = probe.learning
ROOT = learning.ROOT
PROBE_ROOT = ROOT / ".venv/local-support-v2-backward-probe-v1-repair1"
PROBE_ARCHIVE_SHA = "ac60b110ad32c71aaca095128d1f64e087b924e68aca90b4c362422dc830b012"
read, save = probe.read, probe.save


def training_schedule(examples, rows, orders):
    recipe = learning.load_recipe()
    expected = {e["id"]: e for e in examples if e["split"] == "train"}
    if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
        raise ValueError("Prepared training inventory differs from reviewed partition")
    for row in rows:
        if [t["id"] for t in row["targets"]] != [
            c["id"] for c in expected[row["id"]]["request"]["candidates"]
        ]:
            raise ValueError("Prepared training target order changed")
    if orders != learning.epoch_orders(list(expected), recipe):
        raise ValueError("Prepared training shuffle changed")
    schedule = []
    accumulation = recipe["gradient_accumulation"]
    for epoch, order in enumerate(orders, 1):
        for offset in range(0, len(order), accumulation):
            schedule.append(
                dict(
                    update=len(schedule) + 1,
                    epoch=epoch,
                    cases=order[offset : offset + accumulation],
                )
            )
    if len(schedule) != recipe["optimizer_updates"] or any(len(r["cases"]) != 4 for r in schedule):
        raise ValueError("Unexpected training update budget")
    return schedule


def grade_corpus(examples, labels, observations):
    if set(observations) != {e["id"] for e in examples}:
        raise ValueError("Every reviewed corpus case needs an observation")
    grading = learning.load_sibling("joint_support_grading")
    return {
        split: grading.grade_selections(
            examples,
            labels,
            {e["id"]: observations[e["id"]] for e in examples if e["split"] == split},
            split,
        )
        for split in ("train", "tuning", "development")
    }


def check_parity(before, after, expected_cases=123):
    if len(before) != expected_cases or set(before) != set(after):
        raise ValueError("Expected exact deployment parity inventory of 123 frames")
    return learning.load_sibling("joint_support_training").merge_agreement(before, after)


def grade_boundaries(cases, observations):
    expected_ids = {
        "boundary:" + c["id"] for c in cases if c["expected_availability"] == "admitted"
    }
    if set(observations) != expected_ids:
        raise ValueError("Every admitted boundary control needs an observation")
    expected = retained = unavailable = unavailable_supported = unsupported = 0
    outcomes = []
    for case in cases:
        wanted = set(case["supported_target_ids"])
        row = observations.get(
            "boundary:" + case["id"],
            dict(status="unavailable", selected_ids=[], reason="whole_request_exceeds_admission"),
        )
        valid = row.get("status") == "assessed"
        chosen = set(row.get("selected_ids", [])) if valid else set()
        expected += len(wanted)
        retained += len(chosen & wanted)
        unsupported += len(chosen - wanted)
        unavailable += not valid
        unavailable_supported += len(wanted) if not valid else 0
        outcomes.append(dict(id=case["id"], **row))
    return dict(
        cases=len(cases),
        expected_supported=expected,
        retained_supported=retained,
        unavailable=unavailable,
        unavailable_supported=unavailable_supported,
        unsupported=unsupported,
        supported_retention=retained / expected if expected else None,
        repeated_synthetic_controls=True,
        fresh_acceptance=False,
        outcomes=outcomes,
    )


def record_failure(root, result, error):
    for filename, counter in (
        ("microbatches.jsonl", "completed_microbatches"),
        ("updates.jsonl", "completed_updates"),
    ):
        path = root / filename
        if path.exists():
            rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    result[filename + "_partial_record"] = True
            result[counter] = max(result.get(counter, 0), len(rows))
    completed = result.get("completed_updates", 0)
    expected = result.get("expected_updates", 84)
    result.update(
        status="failed",
        error_type=type(error).__name__,
        error=str(error),
        uncompleted_updates=expected - completed,
        unattempted_updates=[
            i for i in range(completed + 1, expected + 1) if i != result.get("active_update")
        ],
    )
    save(root / "result.json", result)


def append(root, name, row):
    with (root / name).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


def supervisor_source():
    original = (ROOT / "benchmarks/native_hosts/windows_owned_job.py").read_bytes()
    if (
        learning.sha(ROOT / "benchmarks/native_hosts/windows_owned_job.py")
        != "faa640786feac9e722365efa6e264c73541d1f1921f69cba6d9c8f8913b9f87a"
        or original.count(b"timeout <= 300") != 1
    ):
        raise ValueError("Exact owned-job controller changed")
    return original.replace(b"timeout <= 300", b"timeout <= 600")


def export_checkpoint(model, root, recipe, copy_tokenizer=True):
    import torch
    from safetensors.torch import save_file

    save_file(
        {n: p.detach().cpu().contiguous() for n, p in model.state_dict().items()},
        str(root / "final-master.safetensors"),
    )
    artifact = root / "deployment"
    artifact.mkdir(exist_ok=False)
    model.backbone.save_pretrained(
        artifact,
        safe_serialization=True,
        state_dict={
            n: p.detach().cpu().to(dtype=torch.float16).contiguous()
            for n, p in model.backbone.state_dict().items()
        },
    )
    # save_pretrained derives dtype from the FP32 live model, so explicitly persist
    # the exported tensor precision without changing the master parameters.
    config = copy.deepcopy(model.backbone.config)
    config.dtype = torch.float16
    config.save_pretrained(artifact)
    save_file(
        {
            n: p.detach().cpu().to(dtype=torch.float16).contiguous()
            for n, p in model.classifier.state_dict().items()
        },
        str(artifact / "support-head.safetensors"),
    )
    save(artifact / "recipe.json", recipe)
    if copy_tokenizer:
        for p in (probe.MODEL_ROOT / "model").iterdir():
            if p.name in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}:
                shutil.copyfile(p, artifact / p.name)
    return artifact


def reload_deployment(artifact, recipe, device="cuda:0"):
    import torch
    from transformers import AutoModel
    from safetensors.torch import load_file

    backbone, info = AutoModel.from_pretrained(
        artifact,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype="auto",
        attn_implementation=recipe["attention"],
        reference_compile=False,
        output_loading_info=True,
    )
    if any(info[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Exported backbone reloaded incorrectly")
    if not all(p.dtype == torch.float16 for p in backbone.parameters()):
        raise ValueError("Deployment configuration must reload as FP16 automatically")
    deployed = (
        learning.make_support_model(backbone, recipe["hidden_size"], recipe).half().to(device)
    )
    deployed.classifier.load_state_dict(load_file(str(artifact / "support-head.safetensors")))
    if not all(
        p.dtype == torch.float16 and bool(torch.isfinite(p).all()) for p in deployed.parameters()
    ):
        raise ValueError("Deployment parameter precision/finiteness failure")
    return deployed


def prepare(root):
    probe.assert_interpreter()
    if not root.resolve().is_relative_to((ROOT / ".venv").resolve()):
        raise ValueError("Private training artifacts must stay in this checkout")
    old = read(PROBE_ROOT / "freeze.json")
    terminal, control = read(PROBE_ROOT / "result.json"), read(
        PROBE_ROOT / "execution-control.json"
    )
    if (
        terminal["status"] != "synthetic_backward_probe_passed"
        or control["status"] != "passed"
        or not control["owned_processes_stopped"]
    ):
        raise ValueError("Passing backward gate is required")
    archive = PROBE_ROOT.with_suffix(".zip")
    if learning.sha(archive) != PROBE_ARCHIVE_SHA:
        raise ValueError("Backward evidence archive changed")
    probe.verify(old["files"])
    if (
        old["recipe"] != learning.load_recipe()
        or old["versions"] != probe.capacity().runtime_versions()
    ):
        raise ValueError("Numerical recipe/runtime changed")
    examples, labels, _ = learning.load_corpus()
    rows = read(PROBE_ROOT / "training-weights.json")
    if rows != learning.training_rows(examples, labels):
        raise ValueError("Reviewed training target weights changed")
    orders = read(PROBE_ROOT / "epoch-orders.json")
    schedule = training_schedule(examples, rows, orders)
    tokenizer, binding = probe.capacity().tokenizer_at(probe.MODEL_ROOT)
    if binding != old["tokenizer_binding"]:
        raise ValueError("Tokenizer changed")
    requests = read(PROBE_ROOT / "deployment-parity-requests.json")
    frames = {}
    admission = learning.load_sibling("local_support_v2_admission")
    for row in requests:
        frame = admission.frame_local_support(row["request"], tokenizer, binding)
        if frame.frame_sha256 != row["frame_sha256"]:
            raise ValueError("Complete prepared frame changed: " + row["id"])
        frames[row["id"]] = asdict(frame)
    controls = read(probe.CONTROL_ROOT / "cases.json")
    for case in controls:
        try:
            admission.frame_local_support(case["request"], tokenizer, binding)
            availability = "admitted"
        except ValueError:
            availability = "unavailable"
        if availability != case["expected_availability"]:
            raise ValueError("Frozen boundary availability changed")
    if len(frames) != 123 or len(requests) != 123:
        raise ValueError("Deployment input inventory changed")
    root.mkdir(parents=True, exist_ok=False)
    for name in ("training-weights.json", "epoch-orders.json", "deployment-parity-requests.json"):
        shutil.copyfile(PROBE_ROOT / name, root / name)
    save(root / "schedule.json", schedule)
    save(root / "frames.json", frames)
    (root / "owned-job-600.py").write_bytes(supervisor_source())
    paths = [
        Path(__file__),
        ROOT / "tests/test_local_support_v2_training.py",
        ROOT / "docs/plans/2026-11-12-local-support-corpus-run-v1.md",
        PROBE_ROOT / "freeze.json",
        PROBE_ROOT / "result.json",
        PROBE_ROOT / "execution-control.json",
        archive,
        Path(__file__).with_name("joint_support_grading.py"),
        Path(__file__).with_name("joint_support_training.py"),
    ]
    paths += list(root.iterdir())
    files = {**old["files"], **{str(p.resolve()): learning.sha(p) for p in paths}}
    freeze = dict(
        schema="local-support-corpus-attempt-v1",
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        recipe=old["recipe"],
        versions=old["versions"],
        python=old["python"],
        tokenizer_binding=binding,
        model_root=old["model_root"],
        files=files,
        expected_updates=84,
        training_cases=56,
        training_targets=138,
        comparison_cases=116,
        parity_cases=123,
        timeout_seconds=600,
        cleanup_seconds=10,
        allocator_limit_bytes=10737418240,
        minimum_free_bytes=11274289152,
        backward_archive_sha256=PROBE_ARCHIVE_SHA,
    )
    save(root / "freeze.json", freeze)
    probe.verify(files)
    return dict(
        status="prepared_no_corpus_computation",
        source_commit=freeze["source_commit"],
        bound_files=len(files),
        freeze_sha256=learning.sha(root / "freeze.json"),
    )


def assess(model, frames, root, name):
    import torch

    model.eval()
    model.backbone.gradient_checkpointing_disable()
    observations = {}
    observations.update(
        {
            identity: dict(status="unavailable", selected_ids=[], reason="not_attempted")
            for identity in frames
        }
    )
    with torch.inference_mode():
        for identity in sorted(frames):
            frame = frames[identity]
            try:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(frame).float().cpu().tolist()
                if len(logits) != len(frame.target_ids) or any(
                    not math.isfinite(v) for v in logits
                ):
                    raise ValueError("Nonfinite or mismatched assessment output")
                row = dict(
                    status="assessed",
                    target_ids=list(frame.target_ids),
                    frame_sha256=frame.frame_sha256,
                    logits=logits,
                    selected_ids=[t for t, v in zip(frame.target_ids, logits) if v >= 0],
                )
            except Exception as error:
                row = dict(
                    status="unavailable",
                    target_ids=list(frame.target_ids),
                    frame_sha256=frame.frame_sha256,
                    error_type=type(error).__name__,
                    error=str(error),
                    selected_ids=[],
                )
                observations[identity] = row
                append(root, name + ".jsonl", dict(id=identity, **row))
                # Preserve the failed observation, but never continue after an operational failure.
                save(root / (name + ".json"), observations)
                raise
            observations[identity] = row
            append(root, name + ".jsonl", dict(id=identity, **row))
    save(root / (name + ".json"), observations)
    return observations


def run(root):
    probe.assert_interpreter()
    if (
        not root.resolve().is_relative_to((ROOT / ".venv").resolve())
        or (root / "attempt.started").exists()
    ):
        raise ValueError("Invalid private root or attempt already consumed")
    freeze = read(root / "freeze.json")
    recipe = learning.load_recipe()
    result = dict(
        status="running",
        expected_updates=84,
        completed_updates=0,
        completed_microbatches=0,
        corpus_training_started=False,
        weights_saved=False,
        active_update=None,
    )
    started = time.perf_counter()
    torch = None

    def stage(name, **details):
        result.update(stage=name, elapsed_seconds=time.perf_counter() - started, **details)
        if torch is not None:
            result.update(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        save(root / "progress.json", result)

    try:
        probe.verify(freeze["files"])
        if recipe != freeze["recipe"] or probe.capacity().runtime_versions() != freeze["versions"]:
            raise ValueError("Frozen recipe or dependency versions changed")
        import torch

        if (
            torch.__version__ != "2.10.0+cu128"
            or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4080"
        ):
            raise ValueError("Unexpected training runtime/device")
        torch.set_num_threads(recipe["cpu_threads"])
        torch.set_num_interop_threads(recipe["cpu_interop_threads"])
        free, total = torch.cuda.mem_get_info(0)
        result.update(free_before_bytes=free, total_device_bytes=total)
        if free < freeze["minimum_free_bytes"]:
            result.update(status="not_started_resource_unavailable", stage="free_memory_precheck")
            save(root / "result.json", result)
            return result
        with (root / "attempt.started").open("x", encoding="utf-8") as handle:
            json.dump(
                dict(freeze_sha256=learning.sha(root / "freeze.json"), free_bytes=free), handle
            )
        torch.cuda.set_per_process_memory_fraction(freeze["allocator_limit_bytes"] / total, 0)
        torch.manual_seed(recipe["seed"])
        torch.cuda.manual_seed_all(recipe["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(recipe["sdpa_backends"]["flash"])
        torch.backends.cuda.enable_mem_efficient_sdp(recipe["sdpa_backends"]["memory_efficient"])
        torch.backends.cuda.enable_math_sdp(recipe["sdpa_backends"]["math"])
        torch.cuda.reset_peak_memory_stats()
        stage("bind_frames")
        from types import SimpleNamespace

        frames = {k: SimpleNamespace(**v) for k, v in read(root / "frames.json").items()}
        examples, labels, _ = learning.load_corpus()
        corpus_ids = {e["id"] for e in examples}
        rows = read(root / "training-weights.json")
        if rows != learning.training_rows(examples, labels) or read(
            root / "schedule.json"
        ) != training_schedule(examples, rows, read(root / "epoch-orders.json")):
            raise ValueError("Prepared training inputs changed")
        weights = {r["id"]: r["targets"] for r in rows}
        if len(frames) != 123 or not corpus_ids <= frames.keys():
            raise ValueError("Exact comparison and parity inventory required")
        stage("load_model")
        model, loading = learning.build_model(probe.MODEL_ROOT, recipe)
        save(root / "loading.json", loading)
        result["initial_head_sha256"] = learning.parameter_digest(model.classifier)
        body_before = learning.parameter_digest(model.backbone)
        head_before = result["initial_head_sha256"]
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        stage("initial_comparison")
        initial = assess(model, {i: frames[i] for i in corpus_ids}, root, "initial")
        save(root / "initial-grading.json", grade_corpus(examples, labels, initial))
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.train()
        if not model.backbone.is_gradient_checkpointing or not all(
            layer.gradient_checkpointing for layer in model.backbone.layers
        ):
            raise ValueError("Training checkpointing must cover every layer")
        calls = dict(backbone=0, layers=[0] * recipe["backbone_layers"])
        hooks = []

        def count_body(_module, _inputs):
            calls["backbone"] += 1

        hooks.append(model.backbone.register_forward_pre_hook(count_body))
        for index, layer in enumerate(model.backbone.layers):

            def count_layer(_module, _inputs, index=index):
                calls["layers"][index] += 1

            hooks.append(layer.register_forward_pre_hook(count_layer))
        optimizer = learning.make_optimizer(model, recipe)
        names = {id(p): n for n, p in model.named_parameters()}
        save(
            root / "optimizer-groups.json",
            [
                dict(
                    name=g["name"],
                    lr=g["lr"],
                    weight_decay=g["weight_decay"],
                    parameters=[
                        dict(
                            name=names[id(p)],
                            shape=list(p.shape),
                            numel=p.numel(),
                            dtype=str(p.dtype),
                        )
                        for p in g["params"]
                    ],
                )
                for g in optimizer.param_groups
            ],
        )
        scaler = torch.amp.GradScaler("cuda", **recipe["scaler"])
        result["corpus_training_started"] = True
        for update in read(root / "schedule.json"):
            number = update["update"]
            stage("training", active_update=number)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for identity in update["cases"]:
                stage("forward", active_case=identity)
                frame, targets = frames[identity], weights[identity]
                if list(frame.target_ids) != [t["id"] for t in targets]:
                    raise ValueError("Gradient target order changed")
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = learning.weighted_micro_loss(
                        model(frame),
                        [t["supported"] for t in targets],
                        [t["weight"] for t in targets],
                        recipe,
                    )
                stage("backward", active_case=identity)
                scaler.scale(loss).backward()
                value = float(loss.detach())
                losses.append(value)
                result["completed_microbatches"] += 1
                append(
                    root,
                    "microbatches.jsonl",
                    dict(
                        index=result["completed_microbatches"],
                        update=number,
                        id=identity,
                        loss=value,
                        tokens=len(frame.input_ids),
                        frame_sha256=frame.frame_sha256,
                    ),
                )
                stage("completed_microbatch", active_case=None)
            stage("unscale_and_clip")
            scaler.unscale_(optimizer)
            activity = learning.clip_gradients(model, recipe)
            stage("optimizer_step")
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() != recipe["scaler"]["init_scale"]:
                raise ValueError("AMP scale changed or step skipped")
            learning.validate_updated_state(model, optimizer, number)
            result["completed_updates"] = number
            result["active_update"] = None
            append(
                root,
                "updates.jsonl",
                dict(
                    **update,
                    losses=losses,
                    gradient_activity=activity,
                    scaler=float(scaler.get_scale()),
                    all_parameters_and_adam_states_finite=True,
                ),
            )
            stage("completed_update")
        if result["completed_updates"] != 84 or result["completed_microbatches"] != 336:
            raise ValueError("Incomplete fixed learning budget")
        if calls["backbone"] != 336 or calls["layers"] != [672] * recipe["backbone_layers"]:
            raise ValueError("Training forward/checkpoint recomputation inventory changed")
        result["training_forward_entries"] = calls
        for hook in hooks:
            hook.remove()
        result.update(
            body_changed=learning.parameter_digest(model.backbone) != body_before,
            head_changed=learning.parameter_digest(model.classifier) != head_before,
        )
        if not result["body_changed"] or not result["head_changed"]:
            raise ValueError("Backbone and classifier must both change")
        optimizer.zero_grad(set_to_none=True)
        del optimizer, scaler, loss
        gc.collect()
        torch.cuda.empty_cache()
        stage("final_master_comparison")
        reference = assess(model, frames, root, "final-master")
        corpus_reference = {i: reference[i] for i in corpus_ids}
        save(root / "final-master-grading.json", grade_corpus(examples, labels, corpus_reference))
        stage("save_final_checkpoint")
        master = root / "final-master.safetensors"
        artifact = export_checkpoint(model, root, recipe)
        result["weights_saved"] = True
        del model
        gc.collect()
        torch.cuda.empty_cache()
        stage("reload_fp16_artifact")
        deployed = reload_deployment(artifact, recipe)
        stage("deployment_parity")
        candidate = assess(deployed, frames, root, "deployment")
        parity = check_parity(reference, candidate)
        save(root / "deployment-parity.json", parity)
        save(
            root / "deployment-grading.json",
            grade_corpus(examples, labels, {i: candidate[i] for i in corpus_ids}),
        )
        save(
            root / "boundary-grading.json",
            grade_boundaries(
                read(probe.CONTROL_ROOT / "cases.json"),
                {i: candidate[i] for i in candidate if i.startswith("boundary:")},
            ),
        )
        artifact_files = {
            str(p.relative_to(root)): learning.sha(p) for p in artifact.iterdir() if p.is_file()
        }
        artifact_files[master.name] = learning.sha(master)
        save(
            root / "artifact-receipt.json",
            dict(files=artifact_files, parity_passed=parity["passed"], active=False),
        )
        if not parity["passed"]:
            raise ValueError("Deployment parity failed; no threshold rescue")
        stage("terminal_bindings")
        probe.verify(freeze["files"])
        if probe.capacity().runtime_versions() != freeze["versions"]:
            raise ValueError("Dependency versions changed")
        if (
            max(result["peak_allocated_bytes"], result["peak_reserved_bytes"])
            > freeze["allocator_limit_bytes"]
        ):
            raise ValueError("Allocator cap exceeded")
        result.update(
            status="corpus_attempt_completed",
            bindings_unchanged=True,
            dependency_versions_unchanged=True,
            deployment_parity_passed=True,
            uncompleted_updates=0,
            semantic_status=read(root / "deployment-grading.json")["development"]["status"],
            fresh_acceptance=False,
            full_retrieval_timing=False,
            live_activation=False,
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


def owned(root):
    """Bound this single execution to one Windows job and preserve failed receipts."""
    import runpy

    root = root.resolve()
    if not root.is_relative_to((ROOT / ".venv").resolve()):
        raise ValueError("Private attempt root required")
    freeze = read(root / "freeze.json")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != freeze["source_commit"] or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT
    ):
        raise ValueError("Expected frozen clean source")
    if (root / "execution-control.json").exists() or (root / "attempt.started").exists():
        raise ValueError("Refusing to overwrite previous execution")
    probe.verify(freeze["files"])
    control_files = {str(root / "freeze.json"): learning.sha(root / "freeze.json")}
    result = dict(status="running", source_commit=head, timeout_seconds=600)
    started = time.perf_counter()
    save(root / "execution-control.json", result)
    inventory_code = "import importlib.metadata as m,json;print(json.dumps(sorted([dict(name=d.metadata['Name'],version=d.version) for d in m.distributions()],key=lambda r:r['name'].lower().replace('_','-'))))"

    def inventory():
        return json.loads(
            subprocess.check_output(
                [str(probe.MODEL_PYTHON), "-I", "-c", inventory_code],
                env=probe.environment(root),
                text=True,
                timeout=15,
            )
        )

    try:
        if inventory() != freeze["versions"]:
            raise ValueError("Frozen dependency inventory changed")
        result["job"] = runpy.run_path(str(root / "owned-job-600.py"))["run_owned_job"](
            [str(probe.MODEL_PYTHON), "-I", str(Path(__file__)), "run", str(root)],
            cwd=ROOT,
            env=probe.environment(root),
            output=root / "job",
            timeout=600,
        )
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        if not (root / "result.json").exists():
            try:
                record_failure(
                    root,
                    read(root / "progress.json") if (root / "progress.json").exists() else {},
                    error,
                )
            except Exception as receipt_error:
                result["failure_receipt_error_type"] = type(receipt_error).__name__
    finally:
        checks = {
            "bindings_unchanged": lambda: probe.verify({**freeze["files"], **control_files})
            is None,
            "head_unchanged": lambda: head
            == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "dependency_versions_unchanged": lambda: inventory() == freeze["versions"],
            "owned_processes_stopped": lambda: read(root / "job/process.json")[
                "active_after_cleanup"
            ]
            == 0,
            "owned_job_bound_satisfied": lambda: read(root / "job/process.json")["elapsed_seconds"]
            <= 610,
        }
        for name, check in checks.items():
            try:
                result[name] = bool(check())
            except BaseException as error:
                result[name] = False
                result[name + "_error_type"] = type(error).__name__
        try:
            child = read(root / "result.json")
            result.update(
                child_terminal_verified=child.get("status")
                in {"corpus_attempt_completed", "not_started_resource_unavailable", "failed"},
                child_status=child.get("status"),
            )
            if child.get("status") == "not_started_resource_unavailable" and all(
                result[k] for k in checks
            ):
                result["status"] = child["status"]
            elif child.get("status") != "corpus_attempt_completed" or not all(
                result[k] for k in checks
            ):
                result["status"] = "failed"
        except Exception as error:
            result.update(
                status="failed",
                child_terminal_verified=False,
                child_result_error_type=type(error).__name__,
            )
        result["elapsed_seconds"] = time.perf_counter() - started
        save(root / "execution-control.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "owned"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = {"prepare": prepare, "run": run, "owned": owned}[args.mode](args.directory.resolve())
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(
        1
        if result["status"] == "failed"
        else 2 if result["status"] == "not_started_resource_unavailable" else 0
    )
