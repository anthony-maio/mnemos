"""Inactive support experiment: one backbone pass and real-target logits only.

Requires the optional torch runtime. No checkpoint loading, threshold selection,
eligibility decisions, or product integration occurs here.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class TargetLogits:
    target_ids: tuple[str, ...]
    frame_sha256: str
    logits: torch.Tensor


class JointSupportHead(nn.Module):
    """Accept a bare decoder backbone, never a language-model vocabulary head."""

    def __init__(self, backbone: nn.Module, hidden_size: int):
        super().__init__()
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError("Invalid hidden size")
        if hasattr(backbone, "lm_head"):
            raise ValueError("Pass the bare backbone, not the causal LM wrapper")
        self.backbone = backbone
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self, frames: list[Any], *, pad_token_id: int, max_tokens: int
    ) -> tuple[TargetLogits, ...]:
        if not frames or type(pad_token_id) is not int or pad_token_id < 0:
            raise ValueError("Expected a nonempty batch and valid padding ID")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("Invalid token budget")
        binding = frames[0].tokenizer_sha256
        for frame in frames:
            count = len(frame.target_ids)
            if (
                not frame.input_ids
                or len(frame.input_ids) > max_tokens
                or count > 5
                or len(set(frame.target_ids)) != count
                or frame.tokenizer_sha256 != binding
                or len(frame.decision_positions) != count
                or len(frame.marker_starts) != count
                or any(type(t) is not int or t < 0 for t in frame.input_ids)
            ):
                raise ValueError("Invalid frame or batch binding/budget")
            previous = -1
            for start, end in zip(frame.marker_starts, frame.decision_positions, strict=True):
                if not (
                    type(start) is int
                    and type(end) is int
                    and previous < start <= end < len(frame.input_ids)
                ):
                    raise ValueError("Invalid decision position")
                previous = end

        counts = [len(frame.target_ids) for frame in frames]
        if not any(counts):
            return tuple(
                TargetLogits(f.target_ids, f.frame_sha256, self.classifier.weight.new_empty((0,)))
                for f in frames
            )

        device = self.classifier.weight.device
        width = max(len(f.input_ids) for f in frames)
        # Right padding preserves every trusted marker position and causal prefix.
        ids = torch.full((len(frames), width), pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        rows, positions = [], []
        for row, frame in enumerate(frames):
            length = len(frame.input_ids)
            ids[row, :length] = torch.tensor(frame.input_ids, device=device)
            mask[row, :length] = 1
            rows.extend([row] * len(frame.target_ids))
            positions.extend(frame.decision_positions)
        output = self.backbone(
            input_ids=ids,
            attention_mask=mask,
            position_ids=torch.arange(width, device=device).unsqueeze(0).expand(len(frames), -1),
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state
        if hidden.shape != (len(frames), width, self.classifier.in_features):
            raise ValueError("Unexpected backbone shape")
        states = hidden[torch.tensor(rows, device=device), torch.tensor(positions, device=device)]
        logits = self.classifier(states).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite support logits; decision unavailable")
        # There are no padded target logits to select or include in a loss.
        pieces = logits.split(counts)
        return tuple(
            TargetLogits(f.target_ids, f.frame_sha256, values)
            for f, values in zip(frames, pieces, strict=True)
        )
