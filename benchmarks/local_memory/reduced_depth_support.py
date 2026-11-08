"""Fixed eight-layer experimental classifier backbone; not model equivalence."""

from torch import nn

RETAINED_DEPTH = 8
ORIGINAL_DEPTH = 28


def retain_fixed_prefix(backbone):
    """Retain original embeddings, blocks 0..7, rotary encoding and final norm.

    Mutates an exclusively owned bare Qwen3Model loaded from the bound checkpoint.
    No attention, precision, context or evidence changes are made here.
    """
    from transformers import Qwen3Model

    if type(backbone) is not Qwen3Model:
        raise ValueError("Expected a bare Qwen3Model")
    if (
        backbone.config.num_hidden_layers != ORIGINAL_DEPTH
        or len(backbone.layers) != ORIGINAL_DEPTH
    ):
        raise ValueError("Expected the original 28-layer backbone")
    if backbone.config.use_sliding_window:
        raise ValueError("Expected full-context attention")
    if any(layer.self_attn.layer_idx != i for i, layer in enumerate(backbone.layers)):
        raise ValueError("Unexpected layer ordering")
    backbone.layers = nn.ModuleList(list(backbone.layers[:RETAINED_DEPTH]))
    backbone.config.num_hidden_layers = RETAINED_DEPTH
    # Keep serialized configuration consistent with the actual retained modules.
    backbone.config.layer_types = list(backbone.config.layer_types[:RETAINED_DEPTH])
    return backbone
