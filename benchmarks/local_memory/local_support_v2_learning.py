"""Shared numerical contracts for the single inactive full-encoder attempt."""

import hashlib
import importlib
import importlib.machinery
import json
import math
from pathlib import Path
import random
import sys
import types

ROOT = Path(__file__).resolve().parents[2]


def load_sibling(name):
    """Load only the bound sibling modules, without the eager benchmark API."""
    package_name = "_mnemos_local_support_v2"
    directory = str(Path(__file__).resolve().parent)
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [directory]
        package.__spec__ = importlib.machinery.ModuleSpec(
            package_name, loader=None, is_package=True
        )
        sys.modules[package_name] = package
    elif sys.modules[package_name].__path__ != [directory]:
        raise ValueError("Unexpected sibling package root")
    return importlib.import_module(package_name + "." + name)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_recipe():
    recipe = json.loads(Path(__file__).with_name("local_support_v2_recipe.json").read_bytes())
    if (
        recipe["training_cases"] % recipe["gradient_accumulation"]
        or recipe["epochs"] * recipe["training_cases"] // recipe["gradient_accumulation"]
        != recipe["optimizer_updates"]
        or recipe["training_cases"] / recipe["gradient_accumulation"]
        != recipe["micro_loss_multiplier"]
    ):
        raise ValueError("Inconsistent fixed training budget")
    return recipe


def load_corpus():
    recipe = load_recipe()
    examples, labels = [], {}
    bindings = dict(receipts={}, files={})
    base = ROOT / "benchmarks/local_memory/data"
    for relative, expected in recipe["corpus_receipts"].items():
        receipt_path = base / relative
        if sha(receipt_path) != expected:
            raise ValueError("Reviewed corpus receipt changed: " + relative)
        receipt = json.loads(receipt_path.read_bytes())
        bindings["receipts"][relative] = expected
        bindings["files"][str(receipt_path.relative_to(ROOT))] = expected
        for filename, digest in receipt["files"].items():
            path = receipt_path.parent / filename
            if sha(path) != digest:
                raise ValueError("Reviewed corpus file changed: " + str(path))
            bindings["files"][str(path.relative_to(ROOT))] = digest
        examples.extend(json.loads((receipt_path.parent / "inputs.json").read_bytes()))
        added = json.loads((receipt_path.parent / "labels.json").read_bytes())
        if labels.keys() & added.keys():
            raise ValueError("Duplicate corpus annotation identity")
        labels.update(added)
    audit = load_sibling("joint_support_data").audit_groups(examples, labels)
    if len(examples) != 116 or sum(len(v["targets"]) for v in labels.values()) != 354:
        raise ValueError("Reviewed corpus inventory changed")
    if audit["review_status"] != {"reviewed": 116}:
        raise ValueError("Unreviewed corpus labels")
    return examples, labels, bindings


def training_rows(examples, labels):
    weights = load_sibling("joint_support_loss").training_weights(examples, labels)
    rows = []
    for example in sorted(examples, key=lambda e: e["id"]):
        if example["split"] != "train":
            continue
        annotation = {t["id"]: t for t in labels[example["id"]]["targets"]}
        rows.append(
            dict(
                id=example["id"],
                targets=[
                    dict(
                        id=c["id"],
                        supported=annotation[c["id"]]["supported"],
                        weight=weights[(example["id"], c["id"])],
                    )
                    for c in example["request"]["candidates"]
                ],
            )
        )
    recipe = load_recipe()
    if len(rows) != recipe["training_cases"] or len(weights) != recipe["training_targets"]:
        raise ValueError("Training inventory changed")
    if not math.isclose(sum(weights.values()), 1, abs_tol=1e-12):
        raise ValueError("Training mass must sum to one")
    return rows


def epoch_orders(identities, recipe=None):
    recipe = recipe or load_recipe()
    if len(identities) != recipe["training_cases"] or len(set(identities)) != len(identities):
        raise ValueError("Expected each training case once")
    generator = random.Random(recipe["seed"])
    orders = []
    for _ in range(recipe["epochs"]):
        order = sorted(identities)
        generator.shuffle(order)
        orders.append(order)
    return orders


def optimizer_groups(model, recipe=None):
    recipe = recipe or load_recipe()
    groups = {}
    for section in ("backbone", "classifier"):
        for decay in (True, False):
            name = section + ("_decay" if decay else "_no_decay")
            groups[name] = dict(
                name=name,
                params=[],
                lr=recipe["optimizer"][section + "_lr"],
                weight_decay=recipe["optimizer"]["weight_decay"] if decay else 0.0,
            )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise ValueError("The fixed full-backbone recipe cannot freeze parameters")
        section = name.split(".")[0]
        if section not in {"backbone", "classifier"}:
            raise ValueError("Unexpected trainable parameter namespace")
        group = section + ("_decay" if parameter.ndim >= 2 else "_no_decay")
        groups[group]["params"].append(parameter)
    result = [g for g in groups.values() if g["params"]]
    parameters = [p for g in result for p in g["params"]]
    if not parameters or len(parameters) != len({id(p) for p in parameters}):
        raise ValueError("Missing or duplicate optimizer parameters")
    return result


def make_optimizer(model, recipe=None):
    import torch

    recipe = recipe or load_recipe()
    settings = recipe["optimizer"]
    return torch.optim.AdamW(
        optimizer_groups(model, recipe),
        betas=tuple(settings["betas"]),
        eps=settings["eps"],
        foreach=settings["foreach"],
        fused=settings["fused"],
        amsgrad=settings["amsgrad"],
        capturable=settings["capturable"],
        differentiable=settings["differentiable"],
        maximize=settings["maximize"],
    )


def weighted_micro_loss(logits, supported, weights, recipe=None):
    import torch

    recipe = recipe or load_recipe()
    if (
        logits.ndim != 1
        or logits.numel() != len(supported)
        or len(supported) != len(weights)
        or not weights
    ):
        raise ValueError("Loss target shape mismatch")
    if any(type(v) is not bool for v in supported) or any(
        not math.isfinite(v) or v <= 0 for v in weights
    ):
        raise ValueError("Invalid target or training mass")
    logits = logits.float()
    expected = logits.new_tensor([float(v) for v in supported])
    mass = logits.new_tensor(weights)
    loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(logits, expected, reduction="none")
        * mass
    ).sum()
    loss = loss * recipe["micro_loss_multiplier"]
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite weighted loss")
    return loss


def make_support_model(backbone, hidden_size, recipe=None):
    import torch

    recipe = recipe or load_recipe()

    class FullSupport(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.classifier = torch.nn.Linear(hidden_size, 1, device="cpu", dtype=torch.float32)
            generator = torch.Generator(device="cpu").manual_seed(
                recipe["head_initialization"]["seed"]
            )
            bound = 1 / math.sqrt(hidden_size)
            torch.nn.init.uniform_(self.classifier.weight, -bound, bound, generator=generator)
            torch.nn.init.uniform_(self.classifier.bias, -bound, bound, generator=generator)

        def forward(self, frame):
            count = len(frame.target_ids)
            length = len(frame.input_ids)
            positions = tuple(frame.decision_positions)
            if (
                count > 5
                or length < 2
                or length > recipe["max_tokens"]
                or len(positions) != count
                or len(set(frame.target_ids)) != count
                or len(set(positions)) != count
                or any(type(p) is not int or not 0 < p < length - 1 for p in positions)
            ):
                raise ValueError("Invalid trusted target frame")
            if not count:
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
            if output.last_hidden_state.shape != (1, length, hidden_size):
                raise ValueError("Backbone hidden-state shape changed")
            index = torch.tensor(positions, device=ids.device, dtype=torch.long)
            return self.classifier(output.last_hidden_state[0, index]).reshape(-1)

    return FullSupport()


def build_model(model_root, recipe=None):
    import torch
    from transformers import AutoModel

    recipe = recipe or load_recipe()
    if sha(model_root / "model/model.safetensors") != recipe["weight_sha256"]:
        raise ValueError("Checkpoint weights changed")
    backbone, loading = AutoModel.from_pretrained(
        model_root / "model",
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float32,
        attn_implementation=recipe["attention"],
        reference_compile=False,
        output_loading_info=True,
    )
    if any(loading[k] for k in ("missing_keys", "mismatched_keys", "error_msgs")) or any(
        not k.startswith(("head.", "classifier.")) for k in loading["unexpected_keys"]
    ):
        raise ValueError("Unexpected checkpoint loading result")
    if (
        backbone.config.num_hidden_layers != recipe["backbone_layers"]
        or backbone.config.hidden_size != recipe["hidden_size"]
        or backbone.config.reference_compile is not False
        or sum(p.numel() for p in backbone.parameters()) != recipe["backbone_parameters"]
    ):
        raise ValueError("Backbone geometry changed")
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if not backbone.is_gradient_checkpointing or not all(
        layer.gradient_checkpointing for layer in backbone.layers
    ):
        raise ValueError("Checkpointing not active across every layer")
    model = make_support_model(backbone, recipe["hidden_size"], recipe).to("cuda:0").train()
    if not all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters()):
        raise ValueError("All parameters must remain trainable float32 masters")
    return model, loading


def parameter_digest(module):
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode())
        digest.update(str(parameter.dtype).encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_summary(model, recipe=None):
    import torch

    recipe = recipe or load_recipe()
    activity = dict(embedding=0.0, classifier=0.0)
    activity.update({"layer_" + str(i): 0.0 for i in range(recipe["backbone_layers"])})
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise ValueError("Missing or nonfinite gradient: " + name)
        value = float(parameter.grad.detach().abs().sum())
        if not math.isfinite(value):
            raise ValueError("Nonfinite gradient activity: " + name)
        if name.startswith("backbone.embeddings."):
            activity["embedding"] += value
        elif name.startswith("backbone.layers."):
            activity["layer_" + name.split(".")[2]] += value
        elif name.startswith("classifier."):
            activity["classifier"] += value
    for name, value in activity.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError("No finite nonzero gradient activity: " + name)
    return activity


def clip_gradients(model, recipe=None):
    import torch

    recipe = recipe or load_recipe()
    before = gradient_summary(model, recipe)
    norm = torch.nn.utils.clip_grad_norm_(
        list(model.parameters()),
        recipe["gradient_clip_norm"],
        error_if_nonfinite=True,
        foreach=False,
    )
    if not bool(torch.isfinite(norm)) or float(norm) <= 0:
        raise ValueError("Invalid clipping norm")
    after = gradient_summary(model, recipe)
    return dict(unclipped_norm=float(norm), before=before, after=after)


def validate_updated_state(model, optimizer, expected_step):
    import torch

    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise ValueError("Nonfinite updated parameter: " + name)
        state = optimizer.state.get(parameter)
        if not state or float(state["step"]) != expected_step:
            raise ValueError("Optimizer step missing or skipped: " + name)
        for key, value in state.items():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite Adam state " + key + ": " + name)


def validate_probe_inventory(calls, recipe=None):
    recipe = recipe or load_recipe()
    if (
        calls["backbone"] != recipe["backward_probe"]["microbatches"]
        or calls["layers"]
        != [recipe["backward_probe"]["layer_forward_entries"]] * recipe["backbone_layers"]
    ):
        raise ValueError("Full forward and checkpoint recomputation inventory mismatch")
