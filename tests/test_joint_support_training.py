"""Objective math, merge criteria and terminal worker handling without training."""

from contextlib import nullcontext
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

if __package__:
    from .test_joint_support_head import load
else:  # unittest discovery with tests as the start directory
    from test_joint_support_head import load
import torch

helper = load("joint_support_training")


class ScalarHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, frames, **kwargs):
        return [
            SimpleNamespace(
                target_ids=f.target_ids, logits=self.weight * torch.tensor(f.coefficients)
            )
            for f in frames
        ]


def problem():
    frames = dict(
        a=SimpleNamespace(target_ids=("x", "y"), coefficients=(1.0, -2.0)),
        b=SimpleNamespace(target_ids=("z",), coefficients=(3.0,)),
    )
    training = [
        dict(
            id="a",
            targets=[
                dict(id="x", supported=True, weight=0.1),
                dict(id="y", supported=False, weight=0.2),
            ],
        ),
        dict(id="b", targets=[dict(id="z", supported=True, weight=0.7)]),
    ]
    return frames, training


class TrainingTests(unittest.TestCase):
    def test_full_partition_matches_analytic_weighted_bce_gradient(self):
        frames, training = problem()
        head = ScalarHead()
        loss = helper.accumulate_partition(
            head, training, frames, 0, SimpleNamespace(scale=lambda loss: loss), nullcontext
        )
        expected_gradient = sum(
            w * x * (1 / (1 + math.exp(-0.2 * x)) - y)
            for w, x, y in [(0.1, 1.0, 1.0), (0.2, -2.0, 0.0), (0.7, 3.0, 1.0)]
        )
        self.assertAlmostEqual(head.weight.grad.item(), expected_gradient, places=6)
        self.assertTrue(math.isfinite(loss) and loss > 0)
        reverse = ScalarHead()
        reverse_loss = helper.accumulate_partition(
            reverse,
            training[::-1],
            frames,
            0,
            SimpleNamespace(scale=lambda loss: loss),
            nullcontext,
        )
        self.assertAlmostEqual(reverse_loss, loss, places=6)
        torch.testing.assert_close(reverse.weight.grad, head.weight.grad)

    def test_duplicate_query_cannot_double_count_mass(self):
        frames, training = problem()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            helper.accumulate_partition(
                ScalarHead(), training + training, frames, 0, None, nullcontext
            )

    def test_partial_partition_is_not_renormalized(self):
        frames, training = problem()
        with self.assertRaisesRegex(ValueError, "mass"):
            helper.accumulate_partition(ScalarHead(), training[:1], frames, 0, None, nullcontext)

    def test_target_reordering_is_rejected(self):
        frames, training = problem()
        training[0]["targets"].reverse()
        with self.assertRaisesRegex(ValueError, "frame order"):
            helper.accumulate_partition(ScalarHead(), training, frames, 0, None, nullcontext)

    def test_nonfinite_loss_is_terminal(self):
        frames, training = problem()
        head = ScalarHead()
        with torch.no_grad():
            head.weight.fill_(float("inf"))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            helper.accumulate_partition(head, training, frames, 0, None, nullcontext)

    def test_merge_rejects_small_threshold_flip_even_for_ineligible_target(self):
        before = dict(
            case=dict(
                status="assessed",
                target_ids=["x"],
                frame_sha256="a",
                logits=[0.0001],
                selected_ids=[],
            )
        )
        after = deepcopy(before)
        after["case"]["logits"] = [-0.0001]
        self.assertFalse(helper.merge_agreement(before, after)["passed"])

    def test_merge_error_bound_and_full_case_coverage(self):
        before = dict(
            case=dict(
                status="assessed",
                target_ids=["x"],
                frame_sha256="a",
                logits=[1.0],
                selected_ids=["x"],
            )
        )
        after = deepcopy(before)
        after["case"]["logits"] = [1.01]
        self.assertTrue(helper.merge_agreement(before, after)["passed"])
        after["case"]["logits"] = [1.021]
        self.assertFalse(helper.merge_agreement(before, after)["passed"])
        with self.assertRaisesRegex(ValueError, "exact same"):
            helper.merge_agreement(before, {})
        after["case"]["logits"] = [float("nan")]
        self.assertFalse(helper.merge_agreement(before, after)["passed"])

    def test_unavailable_merge_case_cannot_disappear(self):
        before = dict(case=dict(status="unavailable", selected_ids=[]))
        self.assertFalse(helper.merge_agreement(before, before)["passed"])

    def test_worker_failure_is_recorded_and_cannot_restart(self):
        worker = load("joint_support_training_run")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            (destination / "freeze.json").write_text(json.dumps(dict(files={})))
            with patch.object(
                worker, "verify", side_effect=KeyboardInterrupt("injected before model")
            ):
                with self.assertRaises(KeyboardInterrupt):
                    worker.run(destination)
            result = json.loads((destination / "result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["corpus_training_started"])
            with self.assertRaises(FileExistsError):
                worker.run(destination)

    def test_timeout_keeps_last_update_and_refuses_restart(self):
        worker = load("joint_support_training_run")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            (destination / "progress.json").write_text(
                json.dumps(dict(stage="training_accumulation", completed_updates=3))
            )
            with patch.object(
                worker.subprocess, "run", side_effect=subprocess.TimeoutExpired("worker", 1800)
            ) as call:
                with self.assertRaises(subprocess.TimeoutExpired):
                    worker.supervise(destination)
                self.assertEqual(call.call_args.kwargs["timeout"], 1800)
                with self.assertRaises(FileExistsError):
                    worker.supervise(destination)
            result = json.loads((destination / "result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["completed_updates"], 3)


if __name__ == "__main__":
    unittest.main()
