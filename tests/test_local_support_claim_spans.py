"""Structural target binding and complete claim pooling without a model run."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unicodedata
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ByteTokenizer:
    cls_token_id, sep_token_id = 256, 257
    all_special_ids = [256, 257]

    def __call__(self, text, **kwargs):
        ids, offsets = [], []
        for i, char in enumerate(text):
            encoded = char.encode("utf-8")
            ids.extend(encoded)
            offsets.extend([(i, i + 1)] * len(encoded))
        return dict(input_ids=ids, offset_mapping=offsets)


def request():
    return dict(
        query="Which cap applies?",
        reference_date="2026-11-12",
        candidates=[
            dict(
                id="a",
                version=2,
                content="The cap is 40 unless vetoed.",
                assertion_status="asserted",
                sources=[
                    dict(
                        id="sa",
                        source_event_id="ea",
                        excerpt="A: cap 40, unless vetoed. B has another source.",
                    )
                ],
            ),
            dict(
                id="b",
                version=1,
                content="The cap is 90.",
                assertion_status="asserted",
                sources=[
                    dict(
                        id="sb",
                        source_event_id="eb",
                        excerpt="B: cap 90. The cap is 40 unless vetoed.",
                    )
                ],
            ),
        ],
    )


class ClaimSpanContracts(unittest.TestCase):
    def module(self):
        path = ROOT / "benchmarks/local_memory/local_support_claim_spans.py"
        self.assertTrue(path.is_file(), "Structural claim-span contract is missing")
        name = "local_support_claim_spans_test"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return sys.modules[name]

    def frame(self, value=None, **kwargs):
        return self.module().frame_claim_spans(
            value or request(), ByteTokenizer(), "a" * 64, **kwargs
        )

    def test_full_payload_and_structural_owned_spans_survive_repeated_text(self):
        value = request()
        value["candidates"][1]["content"] = value["candidates"][0]["content"]
        frame = self.frame(value)
        self.assertEqual(json.loads(frame.payload_json), value)
        self.assertEqual([c.id for c in frame.claims], ["a", "b"])
        self.assertNotEqual(frame.claims[0].characters, frame.claims[1].characters)
        for c, expected in zip(frame.claims, value["candidates"]):
            start, end = c.characters
            self.assertEqual(
                frame.text[start:end], json.dumps(expected["content"], ensure_ascii=False)[1:-1]
            )
            self.assertEqual(c.version, expected["version"])
            self.assertEqual(
                [(s.id, s.source_event_id) for s in c.sources],
                [(s["id"], s["source_event_id"]) for s in expected["sources"]],
            )

    def test_escaped_and_unicode_text_is_bound_without_substring_search(self):
        value = request()
        value["candidates"][0]["content"] = 'Caf\u00e9 "40"\\none\nnot 90'
        value["candidates"][1]["sources"][0]["excerpt"] = (
            '"content":"Caf\u00e9" and ' + value["candidates"][0]["content"]
        )
        frame = self.frame(value)
        self.assertEqual(json.loads(frame.payload_json), value)
        self.assertTrue(frame.claims[0].tokens)
        start, end = frame.claims[0].characters
        self.assertEqual(
            json.loads('"' + frame.text[start:end] + '"'), value["candidates"][0]["content"]
        )

    def test_nfc_offsets_preserve_raw_payload_and_cover_composed_claim(self):
        value = request()
        value["candidates"][0]["content"] = "Cafe\u0301 and \u03b1\u0301 must not deploy."
        frame = self.frame(value)
        self.assertTrue(hasattr(frame, "model_text"), "Explicit normalized offset space is missing")
        self.assertEqual(json.loads(frame.payload_json), value)
        self.assertEqual(frame.model_text, unicodedata.normalize("NFC", frame.text))
        claim = frame.claims[0]
        a, b = claim.characters
        x, y = claim.model_characters
        self.assertEqual(frame.model_text[x:y], unicodedata.normalize("NFC", frame.text[a:b]))
        self.assertTrue(claim.tokens)

    def test_reordering_preserves_identity_and_owned_source_binding(self):
        value = request()
        before = self.frame(value)
        value["candidates"].reverse()
        after = self.frame(value)
        self.assertEqual([c.id for c in after.claims], ["b", "a"])
        self.assertEqual(after.claims[0].sources[0].id, before.claims[1].sources[0].id)
        self.assertNotEqual(before.frame_sha256, after.frame_sha256)

    def test_limits_reject_whole_input_and_empty_cohort_keeps_query(self):
        value = request()
        original = self.frame(value)
        self.assertEqual(self.frame(value, max_tokens=len(original.input_ids)), original)
        with self.assertRaises(ValueError):
            self.frame(value, max_tokens=len(original.input_ids) - 1)
        with self.assertRaises(ValueError):
            self.frame(value, max_bytes=len(original.payload_json.encode("utf-8")) - 1)
        value["candidates"] = []
        empty = self.frame(value)
        self.assertEqual(empty.claims, ())
        self.assertEqual(json.loads(empty.payload_json), value)

    def test_missing_token_coverage_and_cross_claim_mapping_are_rejected(self):
        m = self.module()

        class Missing(ByteTokenizer):
            def __call__(self, text, **kwargs):
                encoded = super().__call__(text, **kwargs)
                start = text.index("unless")
                encoded["offset_mapping"] = [
                    (0, 0) if a >= start and a < start + 6 else (a, b)
                    for a, b in encoded["offset_mapping"]
                ]
                return encoded

        with self.assertRaises(ValueError):
            m.frame_claim_spans(request(), Missing(), "a" * 64)

        class Merged(ByteTokenizer):
            def __call__(self, text, **kwargs):
                return dict(input_ids=[10], offset_mapping=[(0, len(text))])

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            m.frame_claim_spans(request(), Merged(), "a" * 64)

    def test_source_inventory_and_label_smuggling_fail_closed(self):
        value = request()
        value["candidates"][0]["sources"] = [
            dict(id=f"s{i}", source_event_id=f"e{i}", excerpt="Complete source.") for i in range(9)
        ]
        with self.assertRaises(ValueError):
            self.frame(value)

        value = request()
        value["expected_support"] = True
        with self.assertRaises(ValueError):
            self.frame(value)

    def test_long_claim_keeps_its_final_condition_in_the_pool(self):
        value = request()
        value["candidates"][0]["content"] = (
            "Recorded detail. " * 350 + "Do not deploy unless veto is cleared."
        )
        frame = self.frame(value)
        span = frame.claims[0]
        self.assertGreater(len(span.tokens), 5000)
        self.assertEqual(
            frame.text[span.characters[0] : span.characters[1]], value["candidates"][0]["content"]
        )
        expected = set(range(span.characters[0] + 1, span.characters[1] + 1))
        self.assertEqual(set(span.tokens), expected)

    def test_nonfinite_shared_head_output_fails_the_readout_contract(self):
        m = self.module()
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("Requires optional tensor runtime")
        frame = self.frame()
        head = torch.nn.Linear(2, 1)
        with torch.no_grad():
            head.bias.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            m.score_claims(torch.zeros((len(frame.input_ids), 2)), frame, head)

    def test_fp32_pooling_reads_every_claim_token_and_shared_head_preserves_order(self):
        m = self.module()
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("Requires optional tensor runtime")
        frame = self.frame()
        hidden = torch.zeros((len(frame.input_ids), 2), dtype=torch.float16, requires_grad=True)
        with torch.no_grad():
            hidden[list(frame.claims[0].tokens), 0] = 4
            hidden[list(frame.claims[1].tokens), 0] = 9
        pooled = m.pool_claims(hidden, frame)
        self.assertEqual(pooled.dtype, torch.float32)
        self.assertTrue(torch.equal(pooled[:, 0], torch.tensor([4.0, 9.0])))
        head = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1.0, 0.0]]))
        logits = m.score_claims(hidden, frame, head)
        self.assertTrue(torch.equal(logits, torch.tensor([4.0, 9.0])))
        logits.sum().backward()
        self.assertTrue(all(bool((hidden.grad[list(c.tokens), 0] > 0).all()) for c in frame.claims))
        outside = set(range(len(frame.input_ids))) - {t for c in frame.claims for t in c.tokens}
        self.assertEqual(float(hidden.grad[list(outside)].abs().sum()), 0)
        value = request()
        value["candidates"] = []
        empty = self.frame(value)
        self.assertEqual(m.pool_claims(torch.zeros((len(empty.input_ids), 2)), empty).shape, (0, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
