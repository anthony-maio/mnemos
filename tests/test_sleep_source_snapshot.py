"""Extraction authority must refer to the evidence shown to the provider."""

from datetime import datetime, timezone
import json

import pytest

from mnemos.modules.sleep import SleepDaemon
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.storage import InMemoryStore


def source():
    return EpisodicEvent(
        id="source",
        event_fingerprint="source",
        role="assistant",
        content="The export format is now HTML.",
        normalized_content="the export format is now html.",
        scope="project",
        scope_id="project",
        metadata={"user_profile_id": "profile"},
        occurred_at=datetime.now(timezone.utc),
    )


async def test_source_role_cannot_change_to_authorize_a_correction_during_prediction():
    event = source()
    fact = MemoryChunk(
        id="fact",
        content="The export format is PDF.",
        version=4,
        metadata={"user_profile_id": "profile", "scope": "project", "scope_id": "project"},
    )

    class Provider:
        async def predict(self, prompt, **kwargs):
            assert '"role": "assistant"' in prompt
            event.role = "user"
            return json.dumps(
                {
                    "decision": "facts",
                    "facts": [
                        {
                            "content": event.content,
                            "operation": "correct",
                            "target_fact_id": fact.id,
                            "expected_version": 4,
                            "evidence_ids": [event.id],
                        }
                    ],
                }
            )

    with pytest.raises(ValueError, match="Source events changed during extraction"):
        await SleepDaemon(InMemoryStore()).propose_facts([event], Provider(), existing_facts=[fact])


@pytest.mark.parametrize("mutation", ["scope", "content", "append", "remove"])
async def test_no_facts_response_does_not_hide_changed_evidence(mutation):
    event = source()
    events = [event]

    class Provider:
        async def predict(self, prompt, **kwargs):
            if mutation == "scope":
                event.scope_id = "different-project"
            elif mutation == "content":
                event.content = "Changed after the prompt was constructed."
            elif mutation == "append":
                events.append(event.model_copy(update={"id": "extra"}))
            else:
                events.clear()
            return json.dumps(
                {"decision": "no_durable_facts", "facts": [], "reason": "Nothing durable."}
            )

    with pytest.raises(ValueError, match="Source events changed during extraction"):
        await SleepDaemon(InMemoryStore()).propose_facts(events, Provider())
