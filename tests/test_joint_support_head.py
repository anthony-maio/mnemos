"""Run with unittest in the isolated torch runtime; no downloaded weights."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise unittest.SkipTest("Requires the isolated optional torch runtime") from exc


def load(name):
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "local_memory" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


serialize = load("joint_support_frame").serialize_joint_cohort
Head = load("joint_support_head").JointSupportHead


def frame(count):
    return serialize(
        dict(
            query="What is established?",
            reference_date="2026-11-08",
            candidates=[
                dict(
                    id=f"fact-{i}",
                    version=1,
                    content=f"Property {i} holds.",
                    assertion_status="asserted",
                    sources=[],
                )
                for i in range(count)
            ],
        ),
        lambda s: list(s.encode()),
        tokenizer_sha256="a" * 64,
        special_ids=frozenset(),
        max_tokens=4096,
    )


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, **kw):
        self.calls.append(kw)
        # Positions make incorrect gathering observable, even when marker tokens repeat.
        states = kw["position_ids"].float() * self.scale
        return SimpleNamespace(last_hidden_state=states.unsqueeze(-1))


class JointHeadTests(unittest.TestCase):
    def test_mixed_zero_through_five_targets_and_gradient_binding(self):
        probe = Probe()
        head = Head(probe, 1)
        with torch.no_grad():
            head.classifier.weight.fill_(1)
            head.classifier.bias.zero_()
        frames = [frame(i) for i in range(6)]
        result = head(frames, pad_token_id=0, max_tokens=4096)
        self.assertEqual(len(probe.calls), 1)
        call = probe.calls[0]
        self.assertFalse(call["use_cache"])
        self.assertFalse(call["output_hidden_states"])
        self.assertFalse(call["output_attentions"])
        for row, (f, r) in enumerate(zip(frames, result)):
            self.assertEqual(r.target_ids, f.target_ids)
            self.assertEqual(r.frame_sha256, f.frame_sha256)
            self.assertEqual(r.logits.tolist(), list(f.decision_positions))
            self.assertEqual(call["attention_mask"][row].sum().item(), len(f.input_ids))
            self.assertTrue((call["input_ids"][row, len(f.input_ids) :] == 0).all())
        sum(r.logits.sum() for r in result).backward()
        self.assertEqual(probe.scale.grad.item(), sum(sum(f.decision_positions) for f in frames))

    def test_all_empty_skips_backbone(self):
        probe = Probe()
        result = Head(probe, 1)([frame(0), frame(0)], pad_token_id=0, max_tokens=4096)
        self.assertEqual([len(r.logits) for r in result], [0, 0])
        self.assertEqual(probe.calls, [])

    def test_reject_invalid_frames_before_forward(self):
        f = frame(1)
        probe = Probe()
        head = Head(probe, 1)
        for bad in [
            replace(f, decision_positions=(len(f.input_ids),)),
            replace(f, decision_positions=(-1,)),
            replace(f, marker_starts=(False,)),
            replace(f, target_ids=("x", "x")),
        ]:
            with self.subTest(bad=bad.target_ids), self.assertRaises(ValueError):
                head([bad], pad_token_id=0, max_tokens=4096)
        with self.assertRaises(ValueError):
            head([f, replace(f, tokenizer_sha256="b" * 64)], pad_token_id=0, max_tokens=4096)
        with self.assertRaises(ValueError):
            head([f], pad_token_id=0, max_tokens=len(f.input_ids) - 1)
        self.assertEqual(probe.calls, [])

    def test_nonfinite_is_unavailable_and_lm_wrapper_is_rejected(self):
        head = Head(Probe(), 1)
        with torch.no_grad():
            head.classifier.weight.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "unavailable"):
            head([frame(1)], pad_token_id=0, max_tokens=4096)
        wrapper = Probe()
        wrapper.lm_head = nn.Linear(1, 256)
        with self.assertRaisesRegex(ValueError, "bare backbone"):
            Head(wrapper, 1)

    def test_random_tiny_qwen_padding_matches_unbatched_logits(self):
        from transformers import Qwen3Config, Qwen3Model

        torch.manual_seed(19)
        config = Qwen3Config(
            vocab_size=256,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            pad_token_id=0,
            max_position_embeddings=4096,
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        head = Head(Qwen3Model(config), 16).eval()
        frames = [frame(0), frame(1), frame(5)]
        with torch.no_grad():
            together = head(frames, pad_token_id=0, max_tokens=4096)
            for f, expected in zip(frames, together):
                actual = head([f], pad_token_id=0, max_tokens=4096)[0]
                torch.testing.assert_close(actual.logits, expected.logits, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
