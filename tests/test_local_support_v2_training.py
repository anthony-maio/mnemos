"""Accounting and export contracts for the single local learning runner."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import runpy
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TrainingContracts(unittest.TestCase):
    def module(self):
        path = ROOT / "benchmarks/local_memory/local_support_v2_training.py"
        self.assertTrue(path.is_file(), "Single corpus runner is missing")
        name = "local_support_v2_training_test"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return sys.modules[name]

    def test_schedule_contains_only_train_cases_with_bound_target_order(self):
        m = self.module()
        examples, labels, _ = m.learning.load_corpus()
        rows = m.learning.training_rows(examples, labels)
        orders = m.learning.epoch_orders([r["id"] for r in rows])
        schedule = m.training_schedule(examples, rows, orders)
        self.assertEqual(len(schedule), 84)
        self.assertEqual(sum(len(r["cases"]) for r in schedule), 336)
        self.assertEqual([r["update"] for r in schedule], list(range(1, 85)))
        self.assertEqual({c for r in schedule for c in r["cases"]}, {r["id"] for r in rows})
        rows[0]["id"] = next(e["id"] for e in examples if e["split"] != "train")
        with self.assertRaisesRegex(ValueError, "training"):
            m.training_schedule(examples, rows, orders)

    def test_schedule_rejects_repeated_or_reordered_prepared_targets(self):
        m = self.module()
        examples, labels, _ = m.learning.load_corpus()
        rows = m.learning.training_rows(examples, labels)
        orders = m.learning.epoch_orders([r["id"] for r in rows])
        orders[0][0] = orders[0][1]
        with self.assertRaises(ValueError):
            m.training_schedule(examples, rows, orders)

    def test_unavailable_outputs_remain_in_recall_and_abstention_denominators(self):
        m = self.module()
        examples, labels, _ = m.learning.load_corpus()
        observations = {e["id"]: dict(status="unavailable", selected_ids=[]) for e in examples}
        scores = m.grade_corpus(examples, labels, observations)
        self.assertEqual(set(scores), {"train", "tuning", "development"})
        for score in scores.values():
            self.assertEqual(score["metrics"]["supported_retention"], 0)
            self.assertEqual(score["metrics"]["critical_retention"], 0)
            self.assertEqual(score["metrics"]["answerable_recall"], 0)
            self.assertEqual(score["metrics"]["abstention"], 0)
            self.assertFalse(score["gates"]["all_observations_available"])

    def test_export_parity_requires_all_frames_and_rejects_near_zero_flip(self):
        m = self.module()
        left = {
            "x": dict(
                status="assessed",
                target_ids=["a"],
                frame_sha256="a" * 64,
                logits=[0.0001],
                selected_ids=["a"],
            )
        }
        right = {"x": {**left["x"], "logits": [-0.0001], "selected_ids": []}}
        self.assertFalse(m.check_parity(left, right, expected_cases=1)["passed"])
        self.assertTrue(m.check_parity(left, left, expected_cases=1)["passed"])
        with self.assertRaisesRegex(ValueError, "123|inventory"):
            m.check_parity(left, left)

    def test_failed_training_keeps_completed_and_uncompleted_updates(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as directory:
            result = dict(completed_updates=12, expected_updates=84, active_update=13)
            m.record_failure(Path(directory), result, RuntimeError("injected"))
            saved = json.loads((Path(directory) / "result.json").read_bytes())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["completed_updates"], 12)
            self.assertEqual(saved["uncompleted_updates"], 72)
            self.assertEqual(saved["unattempted_updates"], list(range(14, 85)))

    def test_training_supervisor_changes_only_the_timeout_guard(self):
        m = self.module()
        self.assertTrue(
            hasattr(m, "supervisor_source"), "600-second supervisor adaptation is missing"
        )
        original = (ROOT / "benchmarks/native_hosts/windows_owned_job.py").read_bytes()
        adapted = m.supervisor_source()
        self.assertEqual(adapted.replace(b"timeout <= 600", b"timeout <= 300"), original)
        self.assertEqual(adapted.count(b"timeout <= 600"), 1)

    def test_adapted_supervisor_accepts_600_and_stops_its_owned_processes(self):
        m = self.module()
        if os.name != "nt" or importlib.util.find_spec("win32job") is None:
            self.skipTest("Requires the controller Windows runtime")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "supervisor.py"
            source.write_bytes(m.supervisor_source())
            result = runpy.run_path(str(source))["run_owned_job"](
                [sys.executable, "-I", "-S", "-c", 'print("owned-ok")'],
                cwd=root,
                env=m.probe.environment(root),
                output=root / "job",
                timeout=600,
            )
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["active_after_cleanup"], 0)

    def test_fp16_export_keeps_master_precision_and_records_reloadable_dtype(self):
        m = self.module()
        self.assertTrue(hasattr(m, "export_checkpoint"), "Explicit FP16 export contract is missing")
        try:
            import torch
            from transformers import ModernBertConfig, ModernBertModel
        except ModuleNotFoundError:
            self.skipTest("Requires the optional pinned model runtime")
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
        model = m.learning.make_support_model(ModernBertModel(config), 8)
        recipe = m.learning.load_recipe()
        recipe["hidden_size"] = 8
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m.export_checkpoint(model, root, recipe, copy_tokenizer=False)
            exported = json.loads((root / "deployment/config.json").read_bytes())
            self.assertEqual(exported.get("dtype", exported.get("torch_dtype")), "float16")
            self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
            reloaded = m.reload_deployment(root / "deployment", recipe, device="cpu")
            self.assertTrue(all(p.dtype == torch.float16 for p in reloaded.parameters()))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value.half(), reloaded.state_dict()[name]))

    def test_timeout_receipt_reconstructs_microbatches_written_after_progress(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "microbatches.jsonl").write_text(
                '{"index":1}\n{"index":2}\n{"index":', encoding="utf-8"
            )
            result = dict(completed_updates=0, completed_microbatches=0, active_update=1)
            m.record_failure(root, result, TimeoutError("injected"))
            self.assertEqual(result["completed_microbatches"], 2)
            self.assertTrue(result["microbatches.jsonl_partial_record"])
            self.assertEqual(result["unattempted_updates"], list(range(2, 85)))

    def test_unavailable_boundary_controls_stay_in_supported_denominator(self):
        m = self.module()
        self.assertTrue(
            hasattr(m, "grade_boundaries"), "Full boundary denominator accounting is missing"
        )
        cases = m.read(m.probe.CONTROL_ROOT / "cases.json")
        observations = {
            "boundary:" + c["id"]: dict(status="assessed", selected_ids=c["supported_target_ids"])
            for c in cases
            if c["expected_availability"] == "admitted"
        }
        grade = m.grade_boundaries(cases, observations)
        self.assertEqual(grade["expected_supported"], 38)
        self.assertEqual(grade["unavailable"], 4)
        self.assertEqual(grade["unavailable_supported"], 17)
        self.assertEqual(grade["supported_retention"], 21 / 38)


if __name__ == "__main__":
    unittest.main(verbosity=2)
