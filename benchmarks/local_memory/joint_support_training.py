"""Fixed objective accumulation and prospective merge checks, separate from IO."""

import math


def accumulate_partition(head, training, frames, pad_token_id, scaler, autocast):
    import torch

    if len({r["id"] for r in training}) != len(training):
        raise ValueError("Duplicate training query")
    mass = sum(t["weight"] for row in training for t in row["targets"])
    if not math.isclose(mass, 1.0, abs_tol=1e-12):
        raise ValueError("Full-partition loss mass must equal one")
    total = 0.0
    for row in training:
        frame = frames[row["id"]]
        targets = row["targets"]
        if [t["id"] for t in targets] != list(frame.target_ids):
            raise ValueError("Training targets do not bind frame order")
        if any(
            type(t["supported"]) is not bool or not math.isfinite(t["weight"]) or t["weight"] <= 0
            for t in targets
        ):
            raise ValueError("Invalid training target or weight")
        with autocast():
            output = head([frame], pad_token_id=pad_token_id, max_tokens=4096)[0]
            if list(output.target_ids) != list(frame.target_ids):
                raise ValueError("Model target order changed")
            logits = output.logits.float()
            expected = logits.new_tensor([float(t["supported"]) for t in targets])
            weights = logits.new_tensor([t["weight"] for t in targets])
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, expected, reduction="none"
                )
                * weights
            ).sum()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        scaler.scale(loss).backward()
        total += float(loss.detach())
    return total


def merge_agreement(before, after):
    """All targets, including ineligible ones, must keep finite logits/signs."""
    if set(before) != set(after) or not before:
        raise ValueError("Merge comparisons must cover the exact same cases")
    maximum = 0.0
    failures = []
    for identity in before:
        left, right = before[identity], after[identity]
        if (
            left.get("status") != "assessed"
            or right.get("status") != "assessed"
            or left.get("target_ids") != right.get("target_ids")
            or left.get("frame_sha256") != right.get("frame_sha256")
        ):
            failures.append(dict(id=identity, reason="unavailable_or_binding_mismatch"))
            continue
        a, b = left["logits"], right["logits"]
        if (
            len(a) != len(b)
            or len(a) != len(left["target_ids"])
            or any(not math.isfinite(v) for v in a + b)
        ):
            failures.append(dict(id=identity, reason="nonfinite_or_shape_mismatch"))
            continue
        error = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
        maximum = max(maximum, error)
        if (
            error > 0.02
            or any((x >= 0) != (y >= 0) for x, y in zip(a, b))
            or left["selected_ids"] != right["selected_ids"]
        ):
            failures.append(
                dict(id=identity, reason="logit_or_decision_disagreement", max_absolute_error=error)
            )
    return dict(
        passed=not failures, max_absolute_logit_error=maximum, failures=failures, cases=len(before)
    )
