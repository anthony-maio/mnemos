"""Inactive, capability-aware GQA representation for the support experiment.

No global backend toggles or installed-library patches. The private dispatch
query is optional: unavailable/unknown capability uses explicit KV repetition.
"""

import torch
from torch.nn import functional as F


def native_gqa_uses_fused_kernel(query, key, value, *, attn_mask, dropout_p, is_causal, scale):
    if query.device.type != "cuda":
        return False
    chooser = getattr(torch, "_fused_sdp_choice", None)
    if chooser is None:
        return False
    try:
        choice = int(
            chooser(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=True,
            )
        )
    except (RuntimeError, TypeError, ValueError, NotImplementedError):
        return False
    return choice in {
        int(torch.backends.cuda.SDPBackend.FLASH_ATTENTION),
        int(torch.backends.cuda.SDPBackend.EFFICIENT_ATTENTION),
        int(torch.backends.cuda.SDPBackend.CUDNN_ATTENTION),
    }


def grouped_sdpa(
    query, key, value, *, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, observer=None
):
    """Preserve GQA semantics while avoiding an unnecessary math-only layout.

    Returns the output and representation name for explicit probe accounting.
    This helper never predicts support and never shortens a sequence.
    """
    if min(query.ndim, key.ndim, value.ndim) < 3:
        raise ValueError("Expected head, sequence and feature dimensions")
    if torch.is_grad_enabled() and attn_mask is not None and attn_mask.requires_grad:
        raise ValueError("Learned attention masks are outside this adapter's scope")
    q_heads, k_heads, v_heads = query.shape[-3], key.shape[-3], value.shape[-3]
    if not q_heads or not k_heads or k_heads != v_heads or q_heads % k_heads:
        raise ValueError("Invalid grouped query head counts")
    groups = q_heads // k_heads
    kwargs = dict(attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    needs_grad = torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value))
    # The repeated-head fused backward exceeded the declared FP16 tolerance.
    # Retain native grouped semantics for gradient-bearing calls; this is an
    # inference optimization, not a training-kernel optimization.
    native = groups > 1 and (
        needs_grad or native_gqa_uses_fused_kernel(query, key, value, **kwargs)
    )
    representation = (
        "native_grouped_grad"
        if needs_grad and native
        else "native_grouped" if native else "equal_heads"
    )
    if groups > 1 and not native:
        key = key.repeat_interleave(groups, dim=-3)
        value = value.repeat_interleave(groups, dim=-3)
        representation = "repeated_kv"
    if observer is not None:
        observer(representation, query, key, value, attn_mask, is_causal, dropout_p, scale, native)
    return (
        F.scaled_dot_product_attention(query, key, value, enable_gqa=native, **kwargs),
        representation,
    )


def attention_forward(
    module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs
):
    """Qwen prefill adapter; reuse the original SDPA mask construction."""
    if (
        kwargs.get("output_attentions", False)
        or kwargs.get("head_mask") is not None
        or kwargs.get("sliding_window") is not None
        or kwargs.get("use_cache", False)
        or query.shape[-2] != key.shape[-2]
    ):
        raise ValueError("Portable adapter supports full-attention prefill only")
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
    if is_causal is None:
        is_causal = (
            query.shape[-2] > 1 and attention_mask is None and getattr(module, "is_causal", True)
        )
    output, representation = grouped_sdpa(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        is_causal=is_causal,
        scale=scaling,
        observer=getattr(module, "_mnemos_attention_observer", None),
    )
    return output.transpose(1, 2).contiguous(), None


def register_prefill_adapter():
    """Register a distinct opt-in name; never replace the installed SDPA entry."""
    from transformers.modeling_utils import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    name = "mnemos_portable_grouped_sdpa_v1"
    AttentionInterface.register(name, attention_forward)
    AttentionMaskInterface.register(name, sdpa_mask)
    return name
