"""Structural and execution checks on random tiny weights, not model quality."""

import unittest

if __package__:
    from .test_joint_support_head import load
else:  # unittest discovery with tests as the start directory
    from test_joint_support_head import load

import torch
from transformers import Qwen3Config, Qwen3Model

prefix = load("reduced_depth_support")


def tiny():
    return Qwen3Model(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=28,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            attn_implementation="sdpa",
        )
    ).eval()


class FixedPrefixTests(unittest.TestCase):
    def test_exact_prefix_and_execution(self):
        model = tiny()
        layers = list(model.layers)
        embedding, norm, rotary = model.embed_tokens, model.norm, model.rotary_emb
        calls = []
        handles = [
            layer.register_forward_hook(lambda *args, i=i: calls.append(i))
            for i, layer in enumerate(layers)
        ]
        self.assertIs(prefix.retain_fixed_prefix(model), model)
        self.assertEqual(list(model.layers), layers[:8])
        self.assertIs(model.embed_tokens, embedding)
        self.assertIs(model.norm, norm)
        self.assertIs(model.rotary_emb, rotary)
        self.assertEqual(model.config.num_hidden_layers, 8)
        self.assertEqual(len(model.config.layer_types), 8)
        self.assertFalse(any(name.startswith("layers.8.") for name in model.state_dict()))
        with torch.inference_mode():
            output = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
        self.assertEqual(tuple(output.last_hidden_state.shape), (1, 3, 16))
        self.assertEqual(calls, list(range(8)))
        self.assertTrue(torch.isfinite(output.last_hidden_state).all())
        for handle in handles:
            handle.remove()

    def test_rejects_repeat_pruning_without_mutation(self):
        model = prefix.retain_fixed_prefix(tiny())
        layers = list(model.layers)
        with self.assertRaisesRegex(ValueError, "28-layer"):
            prefix.retain_fixed_prefix(model)
        self.assertEqual(list(model.layers), layers)

    def test_rejects_unexpected_order_before_mutation(self):
        model = tiny()
        model.layers[3].self_attn.layer_idx = 4
        with self.assertRaisesRegex(ValueError, "ordering"):
            prefix.retain_fixed_prefix(model)
        self.assertEqual(len(model.layers), 28)

    def test_rejects_windowed_model_before_mutation(self):
        model = tiny()
        model.config.use_sliding_window = True
        with self.assertRaisesRegex(ValueError, "full-context"):
            prefix.retain_fixed_prefix(model)
        self.assertEqual(len(model.layers), 28)

    def test_rejects_other_module(self):
        with self.assertRaisesRegex(ValueError, "bare Qwen3"):
            prefix.retain_fixed_prefix(torch.nn.Linear(1, 1))


if __name__ == "__main__":
    unittest.main()
