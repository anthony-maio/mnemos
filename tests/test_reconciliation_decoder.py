"""Decoder experiments preserve prompts, raw responses and runtime authority."""

from copy import deepcopy
import json

from jsonschema import Draft202012Validator
import pytest

from benchmarks.local_memory.reconciliation_decoder import DecoderProvider
from mnemos.extraction.source_reconciliation import build_reconciliation_schema
from mnemos.extraction.source_relationship import (
    reconcile_source_relationships,
    relationship_schema,
)
from tests.test_source_reconciliation import Provider, candidate, event, target

MODES = ("bounded", "reversed", "json", "none")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_only_decoder_format_changes(mode) -> None:
    source = event("source", "Actually, the marker is maple.")
    item = candidate(source, "correction_or_rejection")
    schema = relationship_schema(build_reconciliation_schema([item], [target()]))
    before = deepcopy(schema)
    raw = {
        "decisions": [
            {
                "candidate_index": 0,
                "relationship": "user_revision",
                "target_fact_id": "fact-1",
                "expected_version": 1,
                "reason": None,
            }
        ]
    }
    provider = Provider(raw)
    adapter = DecoderProvider(provider, mode=mode)
    response = await adapter.predict("exact prompt", format=schema, temperature=0)
    assert response == json.dumps(raw)
    assert schema == before
    prompt, options = provider.calls[0]
    assert prompt == "exact prompt" and options["temperature"] == 0
    if mode == "none":
        assert "format" not in options
    elif mode == "json":
        assert options["format"] == "json"
    else:
        expected = deepcopy(schema)
        if mode == "reversed":
            expected["properties"]["decisions"]["items"]["oneOf"].reverse()
        assert options["format"] == expected
        validator = Draft202012Validator(options["format"])
        assert validator.is_valid(raw)
        bad = deepcopy(raw)
        bad["decisions"][0]["expected_version"] = 99
        assert not validator.is_valid(bad)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("bad_target,bad_version", [("missing", 1), ("fact-1", 99)])
async def test_runtime_rejects_unobserved_target_and_version(mode, bad_target, bad_version) -> None:
    source = event("source", "Actually, the marker is maple.")
    provider = Provider(
        {
            "decisions": [
                {
                    "candidate_index": 0,
                    "relationship": "user_revision",
                    "target_fact_id": bad_target,
                    "expected_version": bad_version,
                    "reason": None,
                }
            ]
        }
    )
    with pytest.raises(ValueError):
        await reconcile_source_relationships(
            [candidate(source, "correction_or_rejection")],
            [source],
            [target()],
            DecoderProvider(provider, mode=mode),
            prompt_style="direct",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_nonuser_cannot_gain_revision_authority(mode) -> None:
    source = event("source", "The marker is maple.", "assistant")
    provider = Provider(
        {
            "decisions": [
                {
                    "candidate_index": 0,
                    "relationship": "user_revision",
                    "target_fact_id": "fact-1",
                    "expected_version": 1,
                    "reason": None,
                }
            ]
        }
    )
    with pytest.raises(ValueError):
        await reconcile_source_relationships(
            [candidate(source, "attributed_history")],
            [source],
            [target()],
            DecoderProvider(provider, mode=mode),
            prompt_style="direct",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_malformed_response_is_not_repaired(mode) -> None:
    class Malformed(Provider):
        async def predict(self, prompt, **kwargs):
            return "```json\n{}\n```"

    source = event("source", "The marker is maple.")
    provider = Malformed({})
    adapter = DecoderProvider(provider, mode=mode)
    schema = relationship_schema(build_reconciliation_schema([candidate(source)], [target()]))
    assert await adapter.predict("prompt", format=schema) == "```json\n{}\n```"
    with pytest.raises(json.JSONDecodeError):
        await reconcile_source_relationships(
            [candidate(source)],
            [source],
            [target()],
            adapter,
            prompt_style="direct",
        )


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown decoder mode"):
        DecoderProvider(Provider({}), mode="unexpected")
