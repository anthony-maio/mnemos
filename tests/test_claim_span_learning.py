"""Fixed-recipe and target-frame contracts without pretrained encoder runs."""

from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


class ClaimLearningContracts(unittest.TestCase):
    def module(self):
        path = ROOT / "benchmarks/local_memory/claim_span_learning.py"
        self.assertTrue(path.is_file(), "Controlled span-learning runner is missing")
        name = "claim_span_learning_test"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, path)
            m = importlib.util.module_from_spec(spec)
            sys.modules[name] = m
            spec.loader.exec_module(m)
        return sys.modules[name]

    def test_numerical_recipe_is_byte_bound_and_unchanged(self):
        m = self.module()
        self.assertEqual(m.numerical_recipe(), m.legacy.learning.load_recipe())
        self.assertEqual(m.protocol()["parity_frames"], 126)
        self.assertEqual(m.protocol()["optimizer_updates"], 84)
        self.assertEqual(m.protocol()["timeout_seconds"], 600)

    def test_external_saved_frame_requires_native_reconstruction_before_use(self):
        m = self.module()
        with self.assertRaisesRegex(ValueError, "trusted"):
            m.TrustedFrame(SimpleNamespace(input_ids=(1, 2)))

    def torch(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("Requires optional tensor runtime")
        return torch

    def test_wrapper_keeps_parameter_names_and_scores_complete_claim_means(self):
        m, torch = self.module(), self.torch()

        class Body(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = torch.nn.Embedding(32, 4)
                self.calls = 0
                with torch.no_grad():
                    self.embeddings.weight.copy_(torch.arange(128).reshape(32, 4))

            def forward(self, input_ids, **kwargs):
                self.calls += 1
                return SimpleNamespace(last_hidden_state=self.embeddings(input_ids))

        base = torch.nn.Module()
        base.backbone = Body()
        base.classifier = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            base.classifier.weight.copy_(torch.tensor([[1.0, 0, 0, 0]]))
        frame = m.spans.ClaimFrame(
            input_ids=(1, 2, 3, 4, 5),
            text="raw",
            model_text="raw",
            normalization="NFC",
            payload_json="{}",
            query=m.spans.TextSpan((0, 1), (0, 1), (1,)),
            reference_date=m.spans.TextSpan((0, 1), (0, 1), (1,)),
            claims=(
                m.spans.ClaimSpan("a", 1, (0, 1), (0, 1), (1, 2), ()),
                m.spans.ClaimSpan("b", 1, (0, 1), (0, 1), (3,), ()),
            ),
            tokenizer_binding="a" * 64,
            frame_sha256="b" * 64,
        )
        wrapped = m.wrap_model(base)
        self.assertEqual(set(wrapped.state_dict()), set(base.state_dict()))
        self.assertEqual(m.TrustedFrame(frame).target_ids, ("a", "b"))
        logits = wrapped(m.TrustedFrame(frame))
        self.assertTrue(torch.equal(logits, torch.tensor([10.0, 16.0])))
        self.assertEqual(base.backbone.calls, 1)
        self.assertEqual(wrapped(m.TrustedFrame(replace(frame, claims=()))).numel(), 0)
        self.assertEqual(base.backbone.calls, 1)
        with self.assertRaisesRegex(ValueError, "trusted"):
            wrapped(SimpleNamespace(**asdict(frame)))

    def test_reframing_rejects_modified_saved_token_masks(self):
        m = self.module()
        # Reuse the deterministic codec only; actual tokenizer reconstruction is
        # required again when preparing the bound experiment.
        spec = importlib.util.spec_from_file_location(
            "claim_span_test_codec", ROOT / "tests/test_local_support_claim_spans.py"
        )
        test = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test)
        tokenizer = test.ByteTokenizer()
        request = test.request()
        frame = m.spans.frame_claim_spans(request, tokenizer, "a" * 64)
        saved = asdict(frame)
        restored = m.restore_frame(request, saved, tokenizer, "a" * 64)
        self.assertEqual(restored.target_ids, ("a", "b"))
        saved["claims"][0]["tokens"] = (2,)
        with self.assertRaisesRegex(ValueError, "frame"):
            m.restore_frame(request, saved, tokenizer, "a" * 64)

    def test_fresh_fp16_reload_keeps_claim_readout_and_master_weights(self):
        m, torch = self.module(), self.torch()
        self.assertTrue(hasattr(m, "reload_model"), "Explicit claim artifact reload is missing")
        from transformers import ModernBertConfig, ModernBertModel

        config = ModernBertConfig(
            vocab_size=64,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=32,
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
            reference_compile=False,
        )
        recipe = m.numerical_recipe()
        recipe["hidden_size"] = 8
        base = m.learning.make_support_model(ModernBertModel(config), 8, recipe)
        model = m.wrap_model(base).eval()
        frame = m.spans.ClaimFrame(
            input_ids=(1, 2, 3, 4, 5),
            text="raw",
            model_text="raw",
            normalization="NFC",
            payload_json="{}",
            query=m.spans.TextSpan((0, 1), (0, 1), (1,)),
            reference_date=m.spans.TextSpan((0, 1), (0, 1), (1,)),
            claims=(
                m.spans.ClaimSpan("a", 1, (0, 1), (0, 1), (1, 2), ()),
                m.spans.ClaimSpan("b", 1, (0, 1), (0, 1), (3,), ()),
            ),
            tokenizer_binding="a" * 64,
            frame_sha256="b" * 64,
        )
        trusted = m.TrustedFrame(frame)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = m.export_model(model, root, recipe, copy_tokenizer=False)
            loaded = m.reload_model(artifact, recipe, device="cpu").eval()
            self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
            self.assertTrue(all(p.dtype == torch.float16 for p in loaded.parameters()))
            self.assertEqual(set(model.state_dict()), set(loaded.state_dict()))
            for name, p in model.state_dict().items():
                self.assertTrue(torch.equal(p.half(), loaded.state_dict()[name]))
            with torch.inference_mode():
                self.assertTrue(
                    torch.allclose(model(trusted), loaded(trusted).float(), atol=0.02, rtol=0)
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
