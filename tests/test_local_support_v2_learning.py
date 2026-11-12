"""Numerical contracts; direct unittest execution needs no pytest dependency."""

import importlib.util
import math
from pathlib import Path
import sys
import json
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LearningContracts(unittest.TestCase):
    def module(self):
        path = ROOT / "benchmarks/local_memory/local_support_v2_learning.py"
        self.assertTrue(path.is_file(), "Shared frozen learning implementation is missing")
        name = "local_support_v2_learning_test"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return sys.modules[name]

    def torch(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("Requires the optional pinned model runtime")
        return torch

    def test_actual_corpus_target_mass_and_reserved_exclusion(self):
        module = self.module()
        examples, labels, bindings = module.load_corpus()
        rows = module.training_rows(examples, labels)
        self.assertEqual(len(examples), 116)
        self.assertEqual(len(rows), 56)
        self.assertEqual(sum(len(row["targets"]) for row in rows), 138)
        self.assertEqual(len(bindings["receipts"]), 3)
        self.assertAlmostEqual(sum(t["weight"] for row in rows for t in row["targets"]), 1)
        for supported in (False, True):
            self.assertAlmostEqual(
                sum(
                    t["weight"]
                    for row in rows
                    for t in row["targets"]
                    if t["supported"] == supported
                ),
                0.5,
            )
        train = {e["id"] for e in examples if e["split"] == "train"}
        self.assertEqual({row["id"] for row in rows}, train)

    def test_epoch_orders_retain_every_case_and_exact_update_budget(self):
        module = self.module()
        identities = [f"case-{n:02d}" for n in range(56)]
        orders = module.epoch_orders(identities)
        self.assertEqual(len(orders), 6)
        self.assertEqual(orders, module.epoch_orders(list(reversed(identities))))
        self.assertEqual(sum(len(order) // 4 for order in orders), 84)
        self.assertTrue(all(sorted(order) == identities for order in orders))

    def test_four_microbatches_preserve_one_mean_sized_objective(self):
        module, torch = self.module(), self.torch()
        logits = torch.nn.Parameter(torch.zeros(3))
        total = 0.0
        for _ in range(4):
            loss = module.weighted_micro_loss(logits, [False, True, False], [1 / (56 * 3)] * 3)
            total += float(loss.detach())
            loss.backward()
        self.assertAlmostEqual(total, math.log(2), places=6)
        self.assertTrue(torch.allclose(logits.grad, torch.tensor([0.5, -0.5, 0.5]) / 3, atol=1e-7))

    def test_optimizer_covers_all_parameters_once_with_fixed_decay_rule(self):
        module, torch = self.module(), self.torch()
        model = torch.nn.Module()
        model.backbone = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.LayerNorm(2))
        model.classifier = torch.nn.Linear(2, 1)
        groups = module.optimizer_groups(model)
        self.assertEqual(len(groups), 4)
        parameters = [p for g in groups for p in g["params"]]
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        self.assertEqual({id(p) for p in parameters}, {id(p) for p in model.parameters()})
        for group in groups:
            self.assertEqual(
                group["lr"], 0.0001 if group["name"].startswith("classifier") else 0.00002
            )
            self.assertTrue(
                all(group["weight_decay"] == (0.01 if p.ndim >= 2 else 0) for p in group["params"])
            )

    def test_trusted_decision_positions_bind_logit_order_and_empty_bypass(self):
        module, torch = self.module(), self.torch()

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(32, 4)
                self.calls = 0
                with torch.no_grad():
                    self.embedding.weight.copy_(torch.arange(128).reshape(32, 4))

            def forward(self, input_ids, **kwargs):
                self.calls += 1
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        body = Backbone()
        model = module.make_support_model(body, 4)
        with torch.no_grad():
            model.classifier.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
            model.classifier.bias.zero_()
        frame = SimpleNamespace(
            input_ids=(1, 2, 3, 4, 5), target_ids=("a", "b"), decision_positions=(2, 3)
        )
        self.assertTrue(torch.equal(model(frame), torch.tensor([12.0, 16.0])))
        self.assertEqual(body.calls, 1)
        empty = SimpleNamespace(input_ids=(1, 2), target_ids=(), decision_positions=())
        self.assertEqual(model(empty).numel(), 0)
        self.assertEqual(body.calls, 1)

    def test_seeded_head_initialization_does_not_depend_on_prior_rng_consumption(self):
        module, torch = self.module(), self.torch()
        first = module.make_support_model(torch.nn.Linear(4, 4), 4)
        torch.rand(103)
        second = module.make_support_model(torch.nn.Linear(4, 4), 4)
        self.assertTrue(torch.equal(first.classifier.weight, second.classifier.weight))
        self.assertTrue(torch.equal(first.classifier.bias, second.classifier.bias))

    def gradient_fixture(self, torch):
        model = torch.nn.Module()
        model.backbone = torch.nn.Module()
        model.backbone.embeddings = torch.nn.Embedding(4, 2)
        model.backbone.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
        model.classifier = torch.nn.Linear(2, 1)
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        return model

    def test_zero_activity_in_one_layer_cannot_pass_gradient_validation(self):
        module, torch = self.module(), self.torch()
        self.assertTrue(
            hasattr(module, "gradient_summary"), "Per-layer gradient validation is missing"
        )
        model = self.gradient_fixture(torch)
        recipe = module.load_recipe()
        recipe["backbone_layers"] = 2
        module.gradient_summary(model, recipe)
        for p in model.backbone.layers[1].parameters():
            p.grad.zero_()
        with self.assertRaisesRegex(ValueError, "layer_1"):
            module.gradient_summary(model, recipe)

    def test_finite_individual_gradients_with_infinite_norm_cannot_be_clipped_to_zero(self):
        module, torch = self.module(), self.torch()
        self.assertTrue(hasattr(module, "clip_gradients"), "Finite clipping validation is missing")
        model = self.gradient_fixture(torch)
        for p in model.parameters():
            p.grad.fill_(1e30)
        recipe = module.load_recipe()
        recipe["backbone_layers"] = 2
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            module.clip_gradients(model, recipe)

    def test_nonfinite_adam_state_cannot_pass_after_a_parameter_change(self):
        module, torch = self.module(), self.torch()
        self.assertTrue(
            hasattr(module, "validate_updated_state"), "Updated-state validation is missing"
        )
        model = self.gradient_fixture(torch)
        optimizer = module.make_optimizer(model)
        optimizer.step()
        module.validate_updated_state(model, optimizer, 1)
        first = next(model.parameters())
        optimizer.state[first]["exp_avg"].fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "state"):
            module.validate_updated_state(model, optimizer, 1)

    def probe(self):
        path = ROOT / "benchmarks/local_memory/local_support_v2_backward_probe.py"
        self.assertTrue(path.is_file(), "Bounded probe implementation is missing")
        name = "local_support_v2_probe_test"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return sys.modules[name]

    def test_probe_help_is_available_without_site_packages_or_model_loading(self):
        self.probe()
        script = ROOT / "benchmarks/local_memory/local_support_v2_backward_probe.py"
        result = subprocess.run(
            [sys.executable, "-S", str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_failed_probe_receipt_keeps_completed_and_uncompleted_microbatches(self):
        module = self.probe()
        self.assertTrue(hasattr(module, "record_failure"), "Probe failure ledger is missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = dict(
                expected_microbatches=4,
                microbatches=[dict(index=1), dict(index=2)],
                active_microbatch=3,
            )
            module.record_failure(root, result, RuntimeError("injected"))
            saved = json.loads((root / "result.json").read_bytes())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["completed_microbatches"], 2)
            self.assertEqual(saved["uncompleted_microbatches"], 2)
            self.assertEqual(saved["unattempted_microbatches"], [4])

    def test_forward_only_inventory_cannot_claim_checkpoint_recomputation(self):
        module = self.module()
        self.assertTrue(
            hasattr(module, "validate_probe_inventory"),
            "Checkpoint recomputation validation is missing",
        )
        module.validate_probe_inventory(dict(backbone=4, layers=[8] * 22))
        with self.assertRaisesRegex(ValueError, "recomputation"):
            module.validate_probe_inventory(dict(backbone=4, layers=[4] * 22))

    def test_malformed_child_evidence_still_produces_failed_terminal_control(self):
        module = self.probe()
        self.assertTrue(
            hasattr(module, "finalize_child_record"), "Guarded child terminal decoding is missing"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("{", encoding="utf-8")
            record = dict(status="passed")
            module.finalize_child_record(record, path, True)
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["child_result_error_type"], "JSONDecodeError")
            self.assertFalse(record["child_terminal_verified"])

    def test_owned_controller_is_a_standalone_source_controlled_entry_point(self):
        path = ROOT / "benchmarks/local_memory/local_support_v2_owned_probe.py"
        self.assertTrue(path.is_file(), "Source-controlled owned probe controller is missing")
        result = subprocess.run(
            [sys.executable, "-S", str(path), "--help"], capture_output=True, text=True, timeout=15
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
