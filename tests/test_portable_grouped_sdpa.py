import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

try:
    import torch
    from torch.nn import functional as F
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise unittest.SkipTest("Requires optional isolated torch runtime") from exc

path = Path(__file__).resolve().parents[1] / "benchmarks/local_memory/portable_grouped_sdpa.py"
spec = importlib.util.spec_from_file_location("portable_grouped_sdpa", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class PortableGroupedTests(unittest.TestCase):
    def compare(self, device, dtype, causal=False, masked=False, gradients=False):
        torch.manual_seed(37)
        q = (
            torch.randn((2, 13, 16, 128), device=device, dtype=dtype)
            .transpose(1, 2)
            .requires_grad_(gradients)
        )
        k = (
            torch.randn((2, 13, 8, 128), device=device, dtype=dtype)
            .transpose(1, 2)
            .requires_grad_(gradients)
        )
        v = (
            torch.randn((2, 13, 8, 128), device=device, dtype=dtype)
            .transpose(1, 2)
            .requires_grad_(gradients)
        )
        mask = None
        if masked:
            mask = torch.ones((2, 1, 13, 13), dtype=torch.bool, device=device).tril()
            mask[:, :, :, -3:] = False
            if masked == "additive":
                mask = torch.zeros_like(mask, dtype=dtype).masked_fill(~mask, float("-inf"))
        options = dict(attn_mask=mask, is_causal=causal, scale=0.19, dropout_p=0.0)
        actual, representation = module.grouped_sdpa(q, k, v, **options)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            expected = F.scaled_dot_product_attention(q, k, v, enable_gqa=True, **options)
        tolerance = 2e-3 if dtype == torch.float16 else 1e-10
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        if gradients:
            upstream = torch.randn_like(actual)
            ga = torch.autograd.grad((actual * upstream).sum(), (q, k, v), retain_graph=True)
            ge = torch.autograd.grad((expected * upstream).sum(), (q, k, v))
            for a, e in zip(ga, ge):
                torch.testing.assert_close(a, e, atol=tolerance, rtol=tolerance)
        return representation

    def test_cpu_outputs_and_gradients_causal_and_padding(self):
        for causal, masked in ((False, False), (True, False), (False, True), (False, "additive")):
            with self.subTest(causal=causal, masked=masked):
                self.assertEqual(
                    self.compare("cpu", torch.float64, causal, masked, True), "native_grouped_grad"
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_half_outputs_and_gradients_causal_and_padding(self):
        for causal, masked in ((False, False), (True, False), (False, True), (False, "additive")):
            with self.subTest(causal=causal, masked=masked):
                self.assertEqual(
                    self.compare("cuda", torch.float16, causal, masked, True), "native_grouped_grad"
                )
                self.compare("cuda", torch.float16, causal, masked, False)

    def test_missing_capability_query_falls_back_without_global_backend_changes(self):
        flags = (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        )
        with patch.object(torch, "_fused_sdp_choice", None):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float64
            self.assertEqual(self.compare(device, dtype, True), "repeated_kv")
        self.assertEqual(
            flags,
            (
                torch.backends.cuda.flash_sdp_enabled(),
                torch.backends.cuda.mem_efficient_sdp_enabled(),
                torch.backends.cuda.math_sdp_enabled(),
            ),
        )

    def test_equal_heads_and_invalid_grouping(self):
        q = torch.randn(1, 3, 4, 8, dtype=torch.float64)
        output, representation = module.grouped_sdpa(q, q, q)
        self.assertEqual(representation, "equal_heads")
        torch.testing.assert_close(output, F.scaled_dot_product_attention(q, q, q))
        with self.assertRaisesRegex(ValueError, "head counts"):
            module.grouped_sdpa(q, q[:, :2], q[:, :2])
        with self.assertRaisesRegex(ValueError, "Learned attention masks"):
            module.grouped_sdpa(
                q, q, q, attn_mask=torch.zeros(4, 4, dtype=torch.float64, requires_grad=True)
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_unknown_private_dispatch_result_does_not_claim_native_fused_execution(self):
        with patch.object(torch, "_fused_sdp_choice", return_value=None):
            self.assertEqual(self.compare("cuda", torch.float16, True), "repeated_kv")

    def test_native_representation_branch_preserves_grouped_heads(self):
        with patch.object(module, "native_gqa_uses_fused_kernel", return_value=True):
            self.assertEqual(
                self.compare("cpu", torch.float64, True, gradients=False), "native_grouped"
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_observer_sees_actual_final_dispatch_inputs(self):
        records = []

        def observe(representation, q, k, v, mask, causal, dropout, scale, gqa):
            choice = torch._fused_sdp_choice(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=dropout,
                is_causal=causal,
                scale=scale,
                enable_gqa=gqa,
            )
            records.append(
                (representation, q.shape[-3], k.shape[-3], v.shape[-3], gqa, int(choice))
            )

        with torch.inference_mode():
            q = torch.randn(1, 16, 13, 128, device="cuda", dtype=torch.float16)
            k = torch.randn(1, 8, 13, 128, device="cuda", dtype=torch.float16)
            v = torch.randn_like(k)
            module.grouped_sdpa(q, k, v, is_causal=True, observer=observe)
        self.assertEqual(len(records), 1)
        name, qh, kh, vh, gqa, choice = records[0]
        self.assertEqual(kh, vh)
        if name == "repeated_kv":
            self.assertEqual((qh, kh, gqa), (16, 16, False))
        else:
            self.assertEqual((qh, kh, gqa), (16, 8, True))
        self.assertIn(choice, (0, 1, 2, 3))

    def test_tiny_qwen_prefill_reuses_masks_and_matches_original(self):
        from copy import deepcopy
        from transformers import Qwen3Config, Qwen3Model
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
        name = module.register_prefill_adapter()
        self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], original_sdpa)
        config = Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            pad_token_id=0,
            attention_dropout=0.0,
        )
        config._attn_implementation = "sdpa"
        baseline = Qwen3Model(config).double().eval()
        candidate = deepcopy(baseline)
        candidate.config._attn_implementation = name
        ids = torch.tensor([[3, 4, 5, 6, 7], [8, 9, 10, 0, 0]])
        for mask in (ids.ne(0).long(), torch.ones_like(ids)):
            observed = []
            candidate.layers[0].self_attn._mnemos_attention_observer = (
                lambda *args: observed.append(args[0])
            )
            with torch.no_grad():
                expected = baseline(
                    input_ids=ids, attention_mask=mask, use_cache=False
                ).last_hidden_state
                actual = candidate(
                    input_ids=ids, attention_mask=mask, use_cache=False
                ).last_hidden_state
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
            self.assertEqual(observed, ["repeated_kv"])
        with self.assertRaisesRegex(ValueError, "prefill"):
            module.attention_forward(
                candidate.layers[0].self_attn,
                torch.ones(1, 4, 1, 8),
                torch.ones(1, 2, 2, 8),
                torch.ones(1, 2, 2, 8),
                None,
            )


if __name__ == "__main__":
    unittest.main()
