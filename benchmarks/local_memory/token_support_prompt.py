"""Explicit single-user Qwen prompt frame for the token-support experiment."""


def render_raw_prompt(prompt: str) -> str:
    """Frame exact bytes for raw generation and tokenizer preflight.

    The retained Qwen template uses ChatML and this disabled-thinking suffix.
    Reject delimiter-bearing data instead of letting it create another role.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be nonempty text")
    if any(marker in prompt for marker in ("<|", "|>", "<think>", "</think>")):
        raise ValueError("Prompt contains reserved model delimiters")
    return (
        "<|im_start|>user\n" + prompt + " /no_think<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
