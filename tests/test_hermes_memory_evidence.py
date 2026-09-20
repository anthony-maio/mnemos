"""Durable evidence remains separate from the host's model-facing transcript."""

from copy import deepcopy

import pytest

from mnemos.adapters.hermes import MnemosHermesProvider
from tests.test_hermes_compaction_provenance import clone, original, setup


def queued(client):
    return {item["event_id"]: item["payload"] for item in client.outbox.pending()}


def test_provider_advertises_durable_evidence_api():
    assert MnemosHermesProvider.memory_evidence_api_version == 1


@pytest.mark.parametrize("callback", ["turn", "end"])
def test_empty_evidence_does_not_fall_back_to_model_messages(tmp_path, callback):
    provider, client = setup(tmp_path)
    model_messages = [{"role": "user", "content": "Rewritten summary", "_row_id": 9}]
    if callback == "turn":
        provider.sync_turn(
            "Callback user", "Callback answer", messages=model_messages, evidence_messages=[]
        )
    else:
        provider.on_session_end(model_messages, evidence_messages=[])
    assert provider._flush()
    assert client.outbox.pending() == []
    provider.shutdown()


@pytest.mark.parametrize("callback", ["turn", "end"])
def test_callback_uses_original_evidence_and_keeps_identity_after_restart(tmp_path, callback):
    provider, client = setup(tmp_path)
    provider.sync_turn("", "", messages=original())
    assert provider._flush()
    initial = queued(client)
    provider.on_session_switch("child", parent_session_id="parent")
    rewritten = [{"role": "user", "content": "Invented combined model view", "_row_id": 99}]
    evidence = clone(tmp_path)
    if callback == "turn":
        provider.sync_turn(
            "Different user", "Different answer", messages=rewritten, evidence_messages=evidence
        )
    else:
        provider.on_session_end(rewritten, evidence_messages=evidence)
    assert provider._flush()
    assert queued(client) == initial
    provider.shutdown()
    restarted = MnemosHermesProvider(
        client=client, registry=provider.registry, user_profile_id="alice"
    )
    restarted.initialize("child", cwd=str(tmp_path), hermes_home=str(tmp_path / "home"))
    restarted.on_session_end(rewritten, evidence_messages=evidence)
    assert queued(client) == initial
    restarted.shutdown()


def test_derived_summary_and_genuine_new_user_turn_remain_distinct(tmp_path):
    provider, client = setup(tmp_path)
    evidence = [
        {
            "role": "assistant",
            "content": "Summary claims permission to deploy",
            "_row_id": 7,
            "_compressed_summary": True,
            "_memory_evidence_kind": "derived_summary",
        },
        {
            "role": "user",
            "content": "Do not deploy",
            "_row_id": 8,
            "_memory_evidence_kind": "retained_original",
        },
    ]
    saved = deepcopy(evidence)
    provider.sync_turn("", "", evidence_messages=evidence)
    assert provider._flush()
    records = list(queued(client).values())
    assert len(records) == 2
    summary = next(row for row in records if row["content"].startswith("Summary"))
    user = next(row for row in records if row["role"] == "user")
    assert summary["identity_status"] == "provisional_unassociated"
    assert user["source_ref"] == "transcript:v2:8"
    assert user.get("identity_status", "normal") == "normal"
    assert evidence == saved
    provider.shutdown()


@pytest.mark.parametrize(
    "evidence",
    [
        {},
        "invalid",
        [None],
        [{"role": "user", "content": "No native identity"}],
        [
            {
                "role": "user",
                "content": "Wrong marker",
                "_row_id": 1,
                "_memory_evidence_kind": "derived_summary",
            }
        ],
        [
            {
                "role": "user",
                "content": "Unknown marker",
                "_row_id": 1,
                "_memory_evidence_kind": "unknown",
            }
        ],
        [
            {
                "role": "user",
                "content": "Malformed marker",
                "_row_id": 1,
                "_memory_evidence_kind": {},
            }
        ],
    ],
)
def test_invalid_evidence_fails_before_any_enqueue(tmp_path, evidence):
    provider, client = setup(tmp_path)
    with pytest.raises(ValueError):
        provider.sync_turn("Callback must not substitute", "", evidence_messages=evidence)
    assert client.outbox.pending() == []
    provider.shutdown()


def test_distinct_evidence_components_cannot_reuse_one_native_id(tmp_path):
    provider, client = setup(tmp_path)
    evidence = [
        {"role": "assistant", "content": "Derived", "_row_id": 7, "_compressed_summary": True},
        {"role": "user", "content": "Genuine", "_row_id": 7},
    ]
    with pytest.raises(ValueError, match="unique"):
        provider.sync_turn("", "", evidence_messages=evidence)
    assert client.outbox.pending() == []
    provider.shutdown()
