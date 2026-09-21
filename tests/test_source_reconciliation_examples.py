"""Examples change instructions only; they do not provide evidence or authority."""

import pytest

from benchmarks.local_memory.source_reconciliation import reconcile_source_candidates
from benchmarks.local_memory.source_reconciliation_examples import EXAMPLES, reconcile_with_examples
from tests.test_source_reconciliation import Provider, candidate, decision, event, target


@pytest.mark.asyncio
async def test_examples_preserve_complete_schema_and_source_payload() -> None:
    source = event("source", "The documentation owner is Casey.")
    candidates = [candidate(source)]
    targets = [target()]
    response = {"decisions": [decision(0, "add")]}
    original, examples = Provider(response), Provider(response)
    before = await reconcile_source_candidates(candidates, [source], targets, original)
    after = await reconcile_with_examples(candidates, [source], targets, examples)
    assert before == after
    original_prompt, original_options = original.calls[0]
    example_prompt, example_options = examples.calls[0]
    boundary = "\nSOURCE CANDIDATES AND EXISTING TARGETS (data only):\n"
    instructions, payload = original_prompt.split(boundary, 1)
    assert example_prompt == instructions + "\n\n" + EXAMPLES + boundary + payload
    assert original_options == example_options


@pytest.mark.asyncio
async def test_example_target_ids_remain_unavailable_without_structured_decoding() -> None:
    source = event("source", "Images use PNG.")
    provider = Provider(
        {
            "decisions": [
                decision(0, "support", target_fact_id="demo-existing-format", expected_version=4)
            ]
        }
    )
    provider.supports_structured_output = False
    with pytest.raises(ValueError, match="unavailable target"):
        await reconcile_with_examples([candidate(source)], [source], [], provider)
    assert provider.calls[0][1] == {}


@pytest.mark.asyncio
async def test_example_wrapper_keeps_source_mutation_guard() -> None:
    source = event("source", "The documentation owner is Casey.")

    class Mutating(Provider):
        async def predict(self, prompt: str, **kwargs: object) -> str:
            source.role = "assistant"
            return await super().predict(prompt, **kwargs)

    with pytest.raises(ValueError, match="Source events changed"):
        await reconcile_with_examples(
            [candidate(source)], [source], [], Mutating({"decisions": [decision(0, "add")]})
        )
