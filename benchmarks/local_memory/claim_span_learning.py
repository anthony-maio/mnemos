"""One controlled claim-readout learning attempt; no live product integration."""

import argparse
from dataclasses import asdict
import gc
import importlib.util
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

_name = "claim_span_learning_contract"
_spec = importlib.util.spec_from_file_location(
    _name, Path(__file__).with_name("local_support_claim_spans.py")
)
spans = importlib.util.module_from_spec(_spec)
sys.modules[_name] = spans
_spec.loader.exec_module(spans)
learning = spans.learning
legacy = learning.load_sibling("local_support_v2_training")
probe = legacy.probe
ROOT = learning.ROOT
AUDIT = ROOT / ".venv/local-support-claim-spans-v1-nfc1"
PARENT = ROOT / ".venv/local-support-v2-corpus-run-v1-repair1"
read, save, append = legacy.read, legacy.save, legacy.append


def protocol():
    return read(Path(__file__).with_name("claim_span_learning_protocol.json"))


def numerical_recipe():
    policy = protocol()
    path = Path(__file__).with_name(policy["numerical_recipe_file"])
    if learning.sha(path) != policy["numerical_recipe_sha256"]:
        raise ValueError("Fixed parent numerical recipe changed")
    return read(path)


class TrustedFrame:
    """Internal adapter; production execution reconstructs it from native inputs."""

    def __init__(self, frame):
        if not isinstance(frame, spans.ClaimFrame):
            raise ValueError("A trusted native serializer frame is required")
        self.frame = frame

    @property
    def target_ids(self):
        return tuple(c.id for c in self.frame.claims)

    def __getattr__(self, name):
        return getattr(self.frame, name)


def restore_frame(request, saved, tokenizer, binding):
    frame = spans.frame_claim_spans(request, tokenizer, binding)
    if json.loads(json.dumps(asdict(frame))) != json.loads(json.dumps(saved)):
        raise ValueError("Saved span frame does not match native reconstruction")
    return TrustedFrame(frame)


def wrap_model(base):
    import torch

    class ClaimSupport(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = base.backbone
            self.classifier = base.classifier

        def forward(self, frame):
            if not isinstance(frame, TrustedFrame):
                raise ValueError("Expected a trusted native span frame")
            if not frame.target_ids:
                return self.classifier.weight.new_empty((0,))
            ids = torch.tensor(
                [frame.input_ids], device=self.classifier.weight.device, dtype=torch.long
            )
            output = self.backbone(
                input_ids=ids,
                attention_mask=torch.ones_like(ids, dtype=torch.bool),
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
            if output.last_hidden_state.shape != (
                1,
                len(frame.input_ids),
                self.classifier.in_features,
            ):
                raise ValueError("Backbone hidden-state shape changed")
            return spans.score_claims(output.last_hidden_state[0], frame, self.classifier)

    return ClaimSupport().train(base.training)


def fresh_model(recipe):
    model, loading = learning.build_model(probe.MODEL_ROOT, recipe)
    return wrap_model(model), loading


def export_model(model, root, recipe, copy_tokenizer=True):
    policy = protocol()
    artifact = legacy.export_checkpoint(model, root, recipe, copy_tokenizer=copy_tokenizer)
    save(
        artifact / "recipe.json",
        dict(schema=policy["schema"], numerical_reference=recipe, execution_protocol=policy),
    )
    save(
        artifact / "readout.json",
        dict(
            schema=policy["representation"],
            readout=policy["readout"],
            source_sha256=learning.sha(Path(spans.__file__)),
            active=False,
        ),
    )
    return artifact


def reload_model(artifact, recipe, device="cuda:0"):
    policy = protocol()
    metadata = read(artifact / "readout.json")
    exported = read(artifact / "recipe.json")
    if (
        metadata["schema"] != policy["representation"]
        or metadata["readout"] != policy["readout"]
        or metadata["source_sha256"] != learning.sha(Path(spans.__file__))
        or exported["numerical_reference"] != recipe
        or exported["execution_protocol"] != policy
    ):
        raise ValueError("Claim artifact readout or numerical metadata changed")
    return wrap_model(legacy.reload_deployment(artifact, recipe, device=device))


def prepare(root):
    probe.assert_interpreter()
    root = root.resolve()
    if not root.is_relative_to((ROOT / ".venv").resolve()):
        raise ValueError("Private experiment root required")
    policy, recipe = protocol(), numerical_recipe()
    audit = read(AUDIT / "receipt.json")
    prior = read(AUDIT / "freeze.json")
    archive = AUDIT.with_suffix(".zip")
    if (
        audit["status"] != "model_free_span_contract_audit_passed"
        or learning.sha(archive) != policy["span_audit_archive_sha256"]
    ):
        raise ValueError("Reviewed span contract evidence changed")
    probe.verify(prior["files"])
    if probe.capacity().runtime_versions() != prior["versions"]:
        raise ValueError("Frozen model environment changed")
    tokenizer, binding = probe.capacity().tokenizer_at(probe.MODEL_ROOT)
    if binding != prior["tokenizer_binding"]:
        raise ValueError("Frozen tokenizer changed")
    examples, labels, _ = learning.load_corpus()
    rows = read(PARENT / "training-weights.json")
    if rows != learning.training_rows(examples, labels):
        raise ValueError("Training partition or weights changed")
    orders = read(PARENT / "epoch-orders.json")
    schedule = legacy.training_schedule(examples, rows, orders)
    if schedule != read(PARENT / "schedule.json"):
        raise ValueError("Complete training schedule changed")
    requests = {e["id"]: e["request"] for e in examples}
    controls = read(probe.CONTROL_ROOT / "cases.json")
    mapped_controls = []
    for c in controls:
        c = dict(c)
        try:
            spans.frame_claim_spans(c["request"], tokenizer, binding)
            c["expected_availability"] = "admitted"
            requests["boundary:" + c["id"]] = c["request"]
        except ValueError:
            c["expected_availability"] = "unavailable"
        mapped_controls.append(c)
    requests["new-boundary:exact-token"] = read(AUDIT / "new-token-boundary-requests.json")["exact"]
    saved = read(AUDIT / "frames.json")
    # Recover the non-gold Unicode fixture's exact preserved request.
    requests["offset-fixture:unicode-repetition"] = json.loads(
        saved["offset-fixture:unicode-repetition"]["payload_json"]
    )
    if set(requests) != set(saved) or len(requests) != policy["parity_frames"]:
        raise ValueError("Exact 126-frame inventory changed")
    for identity, value in requests.items():
        restore_frame(value, saved[identity], tokenizer, binding)
    root.mkdir(parents=True, exist_ok=False)
    for filename in ("training-weights.json", "epoch-orders.json", "schedule.json"):
        shutil.copyfile(PARENT / filename, root / filename)
    for filename, value in [
        ("requests.json", requests),
        ("frames.json", saved),
        ("controls.json", mapped_controls),
    ]:
        save(root / filename, value)
    (root / "owned-job-600.py").write_bytes(legacy.supervisor_source())
    paths = [
        Path(__file__),
        Path(__file__).with_name("claim_span_learning_protocol.json"),
        ROOT / "tests/test_claim_span_learning.py",
        ROOT / "docs/plans/2026-11-12-claim-span-learning-v1.md",
        AUDIT / "freeze.json",
        AUDIT / "receipt.json",
        AUDIT / "frames.json",
        archive,
        PARENT / "training-weights.json",
        PARENT / "epoch-orders.json",
        PARENT / "schedule.json",
    ]
    paths += list(root.iterdir())
    files = {**prior["files"], **{str(p.resolve()): learning.sha(p) for p in paths}}
    freeze = dict(
        schema=policy["schema"],
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        protocol=policy,
        numerical_recipe=recipe,
        versions=prior["versions"],
        tokenizer_binding=binding,
        files=files,
    )
    save(root / "freeze.json", freeze)
    probe.verify(files)
    return dict(
        status="prepared_no_encoder_computation",
        source_commit=freeze["source_commit"],
        bound_files=len(files),
        freeze_sha256=learning.sha(root / "freeze.json"),
    )


def counters(model, layers):
    calls = dict(backbone=0, layers=[0] * layers)

    def body(_module, _inputs):
        calls["backbone"] += 1

    hooks = [model.backbone.register_forward_pre_hook(body)]
    for i, layer in enumerate(model.backbone.layers):

        def count(_module, _inputs, i=i):
            calls["layers"][i] += 1

        hooks.append(layer.register_forward_pre_hook(count))
    return calls, hooks


def optimizer_step(model, optimizer, scaler, recipe, expected):
    scaler.unscale_(optimizer)
    activity = learning.clip_gradients(model, recipe)
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() != recipe["scaler"]["init_scale"]:
        raise ValueError("AMP scale changed or step skipped")
    learning.validate_updated_state(model, optimizer, expected)
    return activity


def run(root):
    probe.assert_interpreter()
    if (
        not root.resolve().is_relative_to((ROOT / ".venv").resolve())
        or (root / "attempt.started").exists()
    ):
        raise ValueError("Invalid root or attempt already consumed")
    freeze = read(root / "freeze.json")
    recipe, policy = numerical_recipe(), protocol()
    result = dict(
        status="running",
        expected_updates=84,
        completed_updates=0,
        completed_microbatches=0,
        active_update=None,
        corpus_training_started=False,
        weights_saved=False,
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
        if (
            recipe != freeze["numerical_recipe"]
            or policy != freeze["protocol"]
            or probe.capacity().runtime_versions() != freeze["versions"]
        ):
            raise ValueError("Protocol/numerical recipe/runtime changed")
        import torch

        if (
            torch.__version__ != "2.10.0+cu128"
            or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4080"
        ):
            raise ValueError("Unexpected model runtime or device")
        torch.set_num_threads(recipe["cpu_threads"])
        torch.set_num_interop_threads(recipe["cpu_interop_threads"])
        free, total = torch.cuda.mem_get_info(0)
        result.update(free_before_bytes=free, total_device_bytes=total)
        if free < policy["minimum_free_bytes"]:
            result.update(status="not_started_resource_unavailable", stage="free_memory_precheck")
            save(root / "result.json", result)
            return result
        with (root / "attempt.started").open("x", encoding="utf-8") as h:
            json.dump(dict(freeze_sha256=learning.sha(root / "freeze.json"), free_bytes=free), h)
        torch.cuda.set_per_process_memory_fraction(policy["allocator_limit_bytes"] / total, 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(recipe["sdpa_backends"]["flash"])
        torch.backends.cuda.enable_mem_efficient_sdp(recipe["sdpa_backends"]["memory_efficient"])
        torch.backends.cuda.enable_math_sdp(recipe["sdpa_backends"]["math"])
        torch.cuda.reset_peak_memory_stats()

        def seed():
            torch.manual_seed(recipe["seed"])
            torch.cuda.manual_seed_all(recipe["seed"])

        stage("native_frame_reconstruction")
        tokenizer, binding = probe.capacity().tokenizer_at(probe.MODEL_ROOT)
        if binding != freeze["tokenizer_binding"]:
            raise ValueError("Tokenizer changed")
        requests, saved = read(root / "requests.json"), read(root / "frames.json")
        frames = {
            i: restore_frame(value, saved[i], tokenizer, binding) for i, value in requests.items()
        }
        if len(frames) != 126:
            raise ValueError("Exact parity inventory required")
        frame = frames[policy["synthetic_prelude_frame"]]
        if (
            len(frame.input_ids) != 8192
            or len(frame.target_ids) != 5
            or any(len(c.sources) != 8 for c in frame.claims)
        ):
            raise ValueError("Complete synthetic prelude frame changed")
        stage("prelude_load")
        seed()
        model, loading = fresh_model(recipe)
        save(root / "loading.json", loading)
        body_before = learning.parameter_digest(model.backbone)
        head_before = learning.parameter_digest(model.classifier)
        if head_before != policy["initial_head_sha256"]:
            raise ValueError("Deterministic head initialization changed")
        calls, hooks = counters(model, 22)
        optimizer = learning.make_optimizer(model, recipe)
        scaler = torch.amp.GradScaler("cuda", **recipe["scaler"])
        optimizer.zero_grad(set_to_none=True)
        prelude = dict(status="running", microbatches=[])
        save(root / "prelude.json", prelude)
        for i in range(1, 5):
            stage("prelude_forward", prelude_microbatch=i)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = learning.weighted_micro_loss(
                    model(frame),
                    recipe["backward_probe"]["targets"],
                    [recipe["backward_probe"]["target_weight"]] * 5,
                    recipe,
                )
            stage("prelude_backward")
            scaler.scale(loss).backward()
            prelude["microbatches"].append(
                dict(
                    index=i,
                    tokens=len(frame.input_ids),
                    loss=float(loss.detach()),
                    frame_sha256=frame.frame_sha256,
                )
            )
            save(root / "prelude.json", prelude)
        stage("prelude_optimizer_step")
        prelude["gradient_activity"] = optimizer_step(model, optimizer, scaler, recipe, 1)
        learning.validate_probe_inventory(calls, recipe)
        prelude.update(
            status="passed",
            calls=calls,
            scaler=float(scaler.get_scale()),
            body_changed=learning.parameter_digest(model.backbone) != body_before,
            head_changed=learning.parameter_digest(model.classifier) != head_before,
        )
        if not prelude["body_changed"] or not prelude["head_changed"]:
            raise ValueError("Disposable update did not change backbone and head")
        save(root / "prelude.json", prelude)
        for hook in hooks:
            hook.remove()
        del hooks, model, optimizer, scaler, loss
        gc.collect()
        torch.cuda.empty_cache()
        stage("reset_original_checkpoint")
        seed()
        model, _ = fresh_model(recipe)
        if (
            learning.parameter_digest(model.backbone) != body_before
            or learning.parameter_digest(model.classifier) != head_before
        ):
            raise ValueError("Synthetic prelude reset did not restore exact initial parameters")
        result.update(
            prelude_passed=True,
            reset_initial_parameter_digests_equal=True,
            initial_head_sha256=head_before,
        )
        examples, labels, _ = learning.load_corpus()
        corpus_ids = {e["id"] for e in examples}
        rows = read(root / "training-weights.json")
        schedule = read(root / "schedule.json")
        if rows != learning.training_rows(examples, labels) or schedule != legacy.training_schedule(
            examples, rows, read(root / "epoch-orders.json")
        ):
            raise ValueError("Prepared training objective or schedule changed")
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        stage("initial_comparison")
        initial = legacy.assess(model, {i: frames[i] for i in corpus_ids}, root, "initial")
        save(root / "initial-grading.json", legacy.grade_corpus(examples, labels, initial))
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.train()
        if not model.backbone.is_gradient_checkpointing or not all(
            l.gradient_checkpointing for l in model.backbone.layers
        ):
            raise ValueError("Checkpointing must cover every layer")
        calls, hooks = counters(model, 22)
        optimizer = learning.make_optimizer(model, recipe)
        scaler = torch.amp.GradScaler("cuda", **recipe["scaler"])
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
        weights = {r["id"]: r["targets"] for r in rows}
        result["corpus_training_started"] = True
        for update in schedule:
            number = update["update"]
            stage("training", active_update=number)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for identity in update["cases"]:
                frame, targets = frames[identity], weights[identity]
                if list(frame.target_ids) != [t["id"] for t in targets]:
                    raise ValueError("Gradient targets no longer bind claim order")
                stage("forward", active_case=identity)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = learning.weighted_micro_loss(
                        model(frame),
                        [t["supported"] for t in targets],
                        [t["weight"] for t in targets],
                        recipe,
                    )
                stage("backward")
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
            stage("optimizer_step")
            activity = optimizer_step(model, optimizer, scaler, recipe, number)
            result.update(completed_updates=number, active_update=None)
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
        if (
            result["completed_updates"] != 84
            or result["completed_microbatches"] != 336
            or calls != dict(backbone=336, layers=[672] * 22)
        ):
            raise ValueError("Fixed training computation inventory changed")
        result.update(
            training_forward_entries=calls,
            body_changed=learning.parameter_digest(model.backbone) != body_before,
            head_changed=learning.parameter_digest(model.classifier) != head_before,
        )
        if not result["body_changed"] or not result["head_changed"]:
            raise ValueError("Expected a full backbone/head gradient update")
        for hook in hooks:
            hook.remove()
        optimizer.zero_grad(set_to_none=True)
        del optimizer, scaler, loss, hooks
        gc.collect()
        torch.cuda.empty_cache()
        stage("final_master_comparison")
        reference = legacy.assess(model, frames, root, "final-master")
        save(
            root / "final-master-grading.json",
            legacy.grade_corpus(examples, labels, {i: reference[i] for i in corpus_ids}),
        )
        stage("export_final_checkpoint")
        artifact = export_model(model, root, recipe)
        result["weights_saved"] = True
        del model
        gc.collect()
        torch.cuda.empty_cache()
        stage("reload_fp16_artifact")
        deployed = reload_model(artifact, recipe)
        stage("deployment_parity")
        candidate = legacy.assess(deployed, frames, root, "deployment")
        parity = legacy.check_parity(reference, candidate, expected_cases=126)
        save(root / "deployment-parity.json", parity)
        grades = legacy.grade_corpus(examples, labels, {i: candidate[i] for i in corpus_ids})
        save(root / "deployment-grading.json", grades)
        save(
            root / "boundary-grading.json",
            legacy.grade_boundaries(
                read(root / "controls.json"),
                {i: candidate[i] for i in candidate if i.startswith("boundary:")},
            ),
        )
        files = {
            str(p.relative_to(root)): learning.sha(p) for p in artifact.iterdir() if p.is_file()
        }
        files["final-master.safetensors"] = learning.sha(root / "final-master.safetensors")
        save(
            root / "artifact-receipt.json",
            dict(files=files, parity_passed=parity["passed"], active=False),
        )
        if not parity["passed"]:
            raise ValueError("Deployment parity failed; no precision or threshold rescue")
        stage("terminal_bindings")
        probe.verify(freeze["files"])
        if probe.capacity().runtime_versions() != freeze["versions"]:
            raise ValueError("Dependency versions changed")
        if (
            max(result["peak_allocated_bytes"], result["peak_reserved_bytes"])
            > policy["allocator_limit_bytes"]
        ):
            raise ValueError("Declared allocator cap exceeded")
        result.update(
            status="claim_span_attempt_completed",
            deployment_parity_passed=True,
            bindings_unchanged=True,
            versions_unchanged=True,
            semantic_status=grades["development"]["status"],
            permutation_groups_consistent=all(
                g["consistent"] for s in grades.values() for g in s["permutations"]
            ),
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
        legacy.record_failure(root, result, error)
        raise


def owned(root):
    root = root.resolve()
    if not root.is_relative_to((ROOT / ".venv").resolve()):
        raise ValueError("Private root required")
    freeze = read(root / "freeze.json")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != freeze["source_commit"] or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT
    ):
        raise ValueError("Expected frozen clean source")
    if (root / "execution-control.json").exists() or (root / "attempt.started").exists():
        raise ValueError("Refusing to overwrite an earlier execution")
    probe.verify(freeze["files"])
    bound = {str(root / "freeze.json"): learning.sha(root / "freeze.json")}
    record = dict(status="running", source_commit=head, timeout_seconds=600)
    save(root / "execution-control.json", record)
    started = time.perf_counter()
    code = "import importlib.metadata as m,json;print(json.dumps(sorted([dict(name=d.metadata['Name'],version=d.version) for d in m.distributions()],key=lambda r:r['name'].lower().replace('_','-'))))"

    def inventory():
        return json.loads(
            subprocess.check_output(
                [str(probe.MODEL_PYTHON), "-I", "-c", code],
                env=probe.environment(root),
                text=True,
                timeout=15,
            )
        )

    try:
        if inventory() != freeze["versions"]:
            raise ValueError("Model runtime inventory changed")
        record["job"] = runpy.run_path(str(root / "owned-job-600.py"))["run_owned_job"](
            [str(probe.MODEL_PYTHON), "-I", str(Path(__file__)), "run", str(root)],
            cwd=ROOT,
            env=probe.environment(root),
            output=root / "job",
            timeout=600,
        )
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        if not (root / "result.json").exists():
            try:
                legacy.record_failure(
                    root,
                    read(root / "progress.json") if (root / "progress.json").exists() else {},
                    error,
                )
            except Exception as receipt_error:
                record["failure_receipt_error_type"] = type(receipt_error).__name__
    finally:
        checks = {
            "bindings_unchanged": lambda: probe.verify({**freeze["files"], **bound}) is None,
            "head_unchanged": lambda: head
            == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "versions_unchanged": lambda: inventory() == freeze["versions"],
            "owned_processes_stopped": lambda: read(root / "job/process.json")[
                "active_after_cleanup"
            ]
            == 0,
            "owned_job_bound_satisfied": lambda: read(root / "job/process.json")["elapsed_seconds"]
            <= 610,
        }
        for name, check in checks.items():
            try:
                record[name] = bool(check())
            except BaseException as error:
                record[name] = False
                record[name + "_error_type"] = type(error).__name__
        try:
            child = read(root / "result.json")
            record.update(
                child_status=child.get("status"),
                child_terminal_verified=child.get("status")
                in {"claim_span_attempt_completed", "failed", "not_started_resource_unavailable"},
            )
            if child.get("status") == "not_started_resource_unavailable" and all(
                record[k] for k in checks
            ):
                record["status"] = child["status"]
            elif child.get("status") != "claim_span_attempt_completed" or not all(
                record[k] for k in checks
            ):
                record["status"] = "failed"
        except Exception as error:
            record.update(
                status="failed",
                child_terminal_verified=False,
                child_result_error_type=type(error).__name__,
            )
        record["elapsed_seconds"] = time.perf_counter() - started
        save(root / "execution-control.json", record)
    return record


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
