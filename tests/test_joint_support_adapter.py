"""Random-weight adapter/backward mechanics; no corpus or checkpoint training."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path
import json
import tempfile
import subprocess

if __package__:
    from .test_joint_support_head import load
    from .test_reduced_depth_support import tiny
else:  # unittest discovery with tests as the start directory
    from test_joint_support_head import load
    from test_reduced_depth_support import tiny

import torch

adapter = load("joint_support_adapter")
prefix = load("reduced_depth_support")
Head = load("joint_support_head").JointSupportHead


def model():
    backbone = prefix.retain_fixed_prefix(tiny())
    adapter.install_adapters(backbone)
    return Head(backbone, 16)


def frame():
    return SimpleNamespace(
        target_ids=("target",),
        input_ids=(1, 2, 3),
        marker_starts=(2,),
        decision_positions=(2,),
        tokenizer_sha256="a" * 64,
        frame_sha256="b" * 64,
    )


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026110808)
        torch.set_num_threads(1)

    def test_zero_initial_update_and_two_stage_gradient_flow(self):
        base = torch.nn.Linear(16, 12)
        inputs = torch.randn(3, 16)
        before = base(inputs).detach()
        layer = adapter.LowRankLinear(base)
        self.assertTrue(torch.equal(layer(inputs), before))
        layer(inputs).square().sum().backward()
        self.assertIsNone(base.weight.grad)
        self.assertEqual(layer.a.grad.count_nonzero(), 0)
        self.assertGreater(layer.b.grad.norm(), 0)
        with torch.no_grad():
            layer.b.add_(layer.b.grad, alpha=-0.001)
        layer.zero_grad(set_to_none=True)
        layer(inputs).square().sum().backward()
        self.assertGreater(layer.a.grad.norm(), 0)

    def test_merge_is_separate_and_matches_float32_linear(self):
        layer = adapter.LowRankLinear(torch.nn.Linear(16, 12))
        with torch.no_grad():
            layer.b.normal_(std=0.01)
        original = layer.base.weight.detach().clone()
        inputs = torch.randn(3, 16)
        merged = layer.merged_linear()
        torch.testing.assert_close(merged(inputs), layer(inputs), rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(original, layer.base.weight))
        self.assertNotEqual(merged.weight.data_ptr(), layer.base.weight.data_ptr())
        self.assertFalse(merged.weight.requires_grad)

    def test_merge_rejects_nonfinite_update(self):
        layer = adapter.LowRankLinear(torch.nn.Linear(16, 12))
        with torch.no_grad():
            layer.b.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            layer.merged_linear()

    def test_repeated_install_rejects_before_mutating_trainable_flags(self):
        head = model()
        before = {n: p.requires_grad for n, p in head.named_parameters()}
        with self.assertRaisesRegex(ValueError, "already adapted"):
            adapter.install_adapters(head.backbone)
        self.assertEqual(before, {n: p.requires_grad for n, p in head.named_parameters()})

    def test_optimizer_rejects_accidentally_unfrozen_original_parameter(self):
        head = model()
        adapter.optimizer_groups(head)
        head.backbone.norm.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen recipe"):
            adapter.optimizer_groups(head)

    def test_optimizer_rejects_cast_trainable_parameters(self):
        head = model().half()
        with self.assertRaisesRegex(ValueError, "float32"):
            adapter.optimizer_groups(head)

    def test_gradient_guard_rejects_weight_decay_only_step(self):
        probe = load("joint_support_backward_probe")
        head = model()
        for p in head.parameters():
            if p.requires_grad:
                p.grad = torch.zeros_like(p)
        with self.assertRaisesRegex(ValueError, "Nonzero"):
            probe.gradient_activity(head)
        head.classifier.weight.grad.fill_(1)
        with self.assertRaisesRegex(ValueError, "Nonzero"):
            probe.gradient_activity(head)
        head.backbone.layers[0].self_attn.q_proj.b.grad.fill_(1)
        self.assertGreater(probe.gradient_activity(head)["adapter_b_gradient_l1"], 0)

    def test_probe_import_failure_has_receipt(self):
        probe = load("joint_support_backward_probe")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "probe"
            with patch.object(
                probe, "load_capacity", side_effect=KeyboardInterrupt("early import")
            ):
                with self.assertRaises(KeyboardInterrupt):
                    probe.run(destination)
            result = json.loads((destination / "result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["stage"], "imports")

    def test_probe_timeout_retains_last_memory_and_prevents_restart(self):
        probe = load("joint_support_backward_probe")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "probe"

            def stalled_worker(*args, **kwargs):
                destination.mkdir()
                (destination / "progress.json").write_text(
                    json.dumps(dict(stage="backward", peak_allocated_bytes=123))
                )
                raise subprocess.TimeoutExpired("worker", 180)

            with patch.object(probe.subprocess, "run", side_effect=stalled_worker) as call:
                with self.assertRaises(subprocess.TimeoutExpired):
                    probe.supervise(destination)
                self.assertEqual(call.call_args.kwargs["timeout"], 180)
                with self.assertRaises(FileExistsError):
                    probe.supervise(destination)
            result = json.loads((destination / "result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["stage"], "backward")
            self.assertEqual(result["peak_allocated_bytes"], 123)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA autocast")
    def test_cuda_scaled_update_changes_only_adapters_and_head(self):
        head = model().to("cuda")
        # Match the intended storage layout: original model half, trainables float.
        for parameter in head.parameters():
            if not parameter.requires_grad:
                parameter.data = parameter.data.half()
        frozen = {n: p.detach().clone() for n, p in head.named_parameters() if not p.requires_grad}
        trainable = {n: p.detach().clone() for n, p in head.named_parameters() if p.requires_grad}
        optimizer = torch.optim.AdamW(
            adapter.optimizer_groups(head), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )
        scaler = torch.amp.GradScaler(
            "cuda", init_scale=128, growth_factor=2, backoff_factor=0.5, growth_interval=2000
        )
        with torch.autocast("cuda", dtype=torch.float16):
            logits = head([frame()], pad_token_id=0, max_tokens=4096)[0].logits
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), torch.ones_like(logits).float()
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        params = [p for p in head.parameters() if p.requires_grad]
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        self.assertTrue(torch.isfinite(norm))
        scaler.step(optimizer)
        scaler.update()
        self.assertEqual(scaler.get_scale(), 128)
        named = dict(head.named_parameters())
        self.assertTrue(
            all(torch.equal(v, named[n]) and named[n].grad is None for n, v in frozen.items())
        )
        self.assertTrue(any(not torch.equal(v, named[n]) for n, v in trainable.items()))
        self.assertGreater(named["backbone.layers.0.self_attn.q_proj.b"].abs().sum(), 0)
        self.assertFalse(torch.equal(trainable["classifier.weight"], head.classifier.weight))

    @unittest.skipUnless(torch.cuda.is_available(), "Probe requires declared CUDA device")
    def test_probe_preparation_failure_is_terminal_and_records_interruption(self):
        probe = load("joint_support_backward_probe")
        for error in (KeyboardInterrupt, RuntimeError):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "probe"
                capacity = SimpleNamespace(
                    tokenizer_setup=lambda: (_ for _ in ()).throw(error("injected before weights")),
                    write=lambda path, value: path.write_text(json.dumps(value)),
                )
                with patch.object(probe, "load_capacity", return_value=capacity):
                    with self.assertRaises(error):
                        probe.run(destination)
                    result = json.loads((destination / "result.json").read_text())
                    self.assertEqual(result["status"], "failed")
                    self.assertIn(error.__name__, result["error"])
                    with self.assertRaises(FileExistsError):
                        probe.run(destination)


if __name__ == "__main__":
    unittest.main()
