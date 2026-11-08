"""Fixed rank-8 q/v adaptation for the inactive eight-layer support experiment."""

import math

import torch
from torch import nn
from torch.nn import functional as F

RANK = 8
ALPHA = 16
DEPTH = 8


class LowRankLinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        if type(base) is not nn.Linear:
            raise ValueError("Expected an unwrapped linear projection")
        self.base = base
        self.a = nn.Parameter(
            torch.empty(RANK, base.in_features, device=base.weight.device, dtype=torch.float32)
        )
        self.b = nn.Parameter(
            torch.zeros(base.out_features, RANK, device=base.weight.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, inputs):
        original = self.base(inputs)
        delta = F.linear(F.linear(inputs.to(self.a.dtype), self.a), self.b)
        return original + (delta * (ALPHA / RANK)).to(original.dtype)

    def merged_linear(self):
        """Return a separate frozen projection; never overwrite training weights."""
        with torch.no_grad(), torch.autocast(device_type=self.a.device.type, enabled=False):
            weight = (self.base.weight.float() + (ALPHA / RANK) * (self.b @ self.a)).to(
                self.base.weight.dtype
            )
            if not torch.isfinite(weight).all():
                raise ValueError("Nonfinite merged adapter weight")
            result = nn.Linear(
                self.base.in_features,
                self.base.out_features,
                bias=self.base.bias is not None,
                device=weight.device,
                dtype=weight.dtype,
            )
            result.weight.copy_(weight)
            if self.base.bias is not None:
                result.bias.copy_(self.base.bias)
            return result.requires_grad_(False)


def install_adapters(backbone):
    """Validate the complete topology before mutating the owned backbone."""
    from transformers import Qwen3Model

    if (
        type(backbone) is not Qwen3Model
        or backbone.config.num_hidden_layers != DEPTH
        or len(backbone.layers) != DEPTH
    ):
        raise ValueError("Expected the fixed eight-layer bare Qwen3Model")
    projections = []
    for i, layer in enumerate(backbone.layers):
        if layer.self_attn.layer_idx != i:
            raise ValueError("Unexpected layer ordering")
        for name in ("q_proj", "v_proj"):
            linear = getattr(layer.self_attn, name)
            if type(linear) is not nn.Linear:
                raise ValueError("Projection is already adapted or has an unexpected type")
            projections.append((layer.self_attn, name, linear))
    backbone.requires_grad_(False)
    for attention, name, linear in projections:
        setattr(attention, name, LowRankLinear(linear))
    return backbone


def optimizer_groups(head):
    """Fail closed if any original parameter was accidentally made trainable."""
    expected = {
        f"backbone.layers.{i}.self_attn.{p}.{part}"
        for i in range(DEPTH)
        for p in ("q_proj", "v_proj")
        for part in ("a", "b")
    }
    classifier = {"classifier.weight", "classifier.bias"}
    named = dict(head.named_parameters())
    actual = {name for name, p in named.items() if p.requires_grad}
    if actual != expected | classifier:
        raise ValueError("Trainable parameters differ from the frozen recipe")
    if any(named[name].dtype != torch.float32 for name in actual):
        raise ValueError("Trainable parameters must remain float32")
    return [
        dict(params=[named[n] for n in sorted(expected)], lr=0.0001),
        dict(params=[named[n] for n in sorted(classifier)], lr=0.001),
    ]
