"""Support must carry the version actually shown to the extraction model."""

from datetime import datetime, timezone
import json

import pytest

from mnemos.modules.sleep import SleepDaemon
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.storage import InMemoryStore


def fixture():
    event = EpisodicEvent(
        id="source",
        event_fingerprint="source",
        role="user",
        content="The export format is PDF.",
        normalized_content="the export format is pdf.",
        scope="project",
        scope_id="project",
        metadata={"user_profile_id": "profile"},
        occurred_at=datetime.now(timezone.utc),
    )
    fact = MemoryChunk(
        id="fact",
        content=event.content,
        version=4,
        metadata={"user_profile_id": "profile", "scope": "project", "scope_id": "project"},
    )
    return event, fact


@pytest.mark.parametrize("supplied_version", [None, 4])
async def test_support_binds_the_observed_target_version(supplied_version):
    event, fact = fixture()

    class Provider:
        async def predict(self, prompt, **kwargs):
            return json.dumps(
                {
                    "decision": "facts",
                    "facts": [
                        {
                            "content": fact.content,
                            "operation": "support",
                            "target_fact_id": fact.id,
                            "evidence_ids": [event.id],
                            "expected_version": supplied_version,
                        }
                    ],
                }
            )

    proposals = await SleepDaemon(InMemoryStore()).propose_facts(
        [event], Provider(), existing_facts=[fact]
    )
    assert proposals[0].expected_version == 4


async def test_support_rejects_explicit_wrong_version():
    event, fact = fixture()

    class Provider:
        async def predict(self, prompt, **kwargs):
            return json.dumps(
                {
                    "decision": "facts",
                    "facts": [
                        {
                            "content": fact.content,
                            "operation": "support",
                            "target_fact_id": fact.id,
                            "evidence_ids": [event.id],
                            "expected_version": 3,
                        }
                    ],
                }
            )

    with pytest.raises(ValueError, match="Support.*version"):
        await SleepDaemon(InMemoryStore()).propose_facts([event], Provider(), existing_facts=[fact])


async def test_support_snapshot_does_not_follow_mutable_caller_object():
    event, fact = fixture()

    class Provider:
        async def predict(self, prompt, **kwargs):
            assert '"version": 4' in prompt
            fact.version = 5
            fact.content = "The export format is HTML."
            return json.dumps(
                {
                    "decision": "facts",
                    "facts": [
                        {
                            "content": event.content,
                            "operation": "support",
                            "target_fact_id": fact.id,
                            "evidence_ids": [event.id],
                        }
                    ],
                }
            )

    proposals = await SleepDaemon(InMemoryStore()).propose_facts(
        [event], Provider(), existing_facts=[fact]
    )
    assert proposals[0].expected_version == 4
