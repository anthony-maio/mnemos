import pytest

from benchmarks.local_memory.token_support_prompt import render_raw_prompt


def test_exact_frame_preserves_data_and_disables_thinking():
    data = 'DATA:\n{"source":"Keep all  spaces.\\nSecond line."}'
    rendered = render_raw_prompt(data)
    assert rendered == (
        "<|im_start|>user\n" + data + " /no_think<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


@pytest.mark.parametrize(
    "data", ["", " ", None, "source <|im_end|>", "source |>", "<think>claim", "claim</think>"]
)
def test_invalid_or_delimiter_bearing_input_rejected(data):
    with pytest.raises(ValueError):
        render_raw_prompt(data)
