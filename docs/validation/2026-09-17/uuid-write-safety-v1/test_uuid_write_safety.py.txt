"""Opaque UUID digits must not randomly reject durable writes."""

from uuid import UUID

import pytest

from mnemos.client import Outbox
from mnemos.config import MemorySafetyConfig
from mnemos.memory_safety import MemoryWriteFirewall
from tests.test_memory_service import service

IDENTIFIER = "abcdefab-1234-4567-8123-123456789012"


@pytest.mark.parametrize("action", ["redact", "block", "allow"])
@pytest.mark.parametrize(
    "identifier",
    [IDENTIFIER, IDENTIFIER.upper(), "abcdefab-1234-7567-8123-123456789012"],
)
def test_uuid_identifier_is_not_numeric_personal_data(action, identifier):
    firewall = MemoryWriteFirewall(MemorySafetyConfig(pii_action=action))
    text = f"Session {{{identifier}}} remains available."
    decision = firewall.apply(text)
    assert decision.allowed
    assert decision.content == text
    assert decision.matches == ()


@pytest.mark.parametrize("action", ["redact", "block"])
@pytest.mark.parametrize("outside_first", [True, False])
def test_uuid_exception_does_not_cover_adjacent_personal_data(action, outside_first):
    firewall = MemoryWriteFirewall(MemorySafetyConfig(pii_action=action))
    personal = "person@example.com card 4111 1111 1111 1111 phone 212-555-0123 SSN 123-45-6789"
    text = f"{personal}; id {IDENTIFIER}" if outside_first else f"id {IDENTIFIER}; {personal}"
    decision = firewall.apply(text)
    if action == "block":
        assert not decision.allowed
    else:
        assert decision.allowed
        assert IDENTIFIER in decision.content
        for value in ("person@example.com", "4111 1111 1111 1111", "212-555-0123", "123-45-6789"):
            assert value not in decision.content
        assert {m.label for m in decision.matches} == {"email", "credit_card", "phone", "ssn"}
        assert all(m.count == 1 for m in decision.matches)


@pytest.mark.parametrize("prefix", ["api_key=", "token: ", "password="])
def test_uuid_value_does_not_bypass_credential_assignment(prefix):
    assert not MemoryWriteFirewall().apply(prefix + IDENTIFIER).allowed


@pytest.mark.parametrize(
    "text",
    [
        "abcdefab-1234-0567-8123-123456789012",  # Not a recognized UUID version.
        "abcdefab-1234-4567-7123-123456789012",  # Not the RFC UUID variant.
        "xyz-8123-123456789012",
        "id " + IDENTIFIER + " 4111111111111111",
        "prefix_" + IDENTIFIER,
        "prefix-" + IDENTIFIER,
        "A" + IDENTIFIER,
        "12345678-1234-4567-8123-abcdefabcdefZ",
        "abcdefab-1234-4567-8abc-123456789012 3456",  # Numeric match crosses UUID boundary.
    ],
)
def test_non_uuid_numeric_data_keeps_existing_redaction(text):
    decision = MemoryWriteFirewall().apply(text)
    assert decision.content != text
    assert any(match.label == "credit_card" for match in decision.matches)


def test_uuid_source_identity_survives_outbox_round_trip(tmp_path):
    outbox = Outbox(tmp_path / "outbox.json")
    payload = {"event_id": IDENTIFIER, "content": "Use cedar-otter labels."}
    outbox.append(payload)
    assert outbox.pending()[0]["payload"]["event_id"] == IDENTIFIER


@pytest.mark.asyncio
async def test_generated_uuid_remember_and_correction_commit_with_replay(tmp_path, monkeypatch):
    import mnemos.memory_service as service_module

    memory, context = service(tmp_path)
    for name in type(memory.engine.config.modules).model_fields:
        setattr(memory.engine.config.modules, name, False)
    identifiers = iter([UUID(IDENTIFIER), UUID("abcdefab-1234-4567-8234-123456789012")])
    monkeypatch.setattr(service_module, "uuid4", lambda: next(identifiers))
    try:
        remembered = await memory.dispatch(
            "remember", {"context": context.model_dump(), "content": "Use cedar-otter labels."}
        )
        assert remembered["status"] == "committed", remembered
        assert await memory.maintenance.run_once(force=True) == 1
        target = memory.engine.store.get_all()[0]
        corrected = await memory.dispatch(
            "correct",
            {
                "context": context.model_dump(),
                "fact_id": target.id,
                "expected_version": 1,
                "content": "Use maple-heron labels.",
            },
        )
        assert corrected["status"] == "committed", corrected
        assert await memory.maintenance.run_once(force=True) == 1
        assert memory.engine.store.get(target.id).content == "Use maple-heron labels."
        replay = await memory.dispatch(
            "remember",
            {
                "context": context.model_dump(),
                "event_id": IDENTIFIER,
                "content": "Use cedar-otter labels.",
            },
        )
        assert replay["event_id"] == remembered["event_id"]
        assert replay["processing_status"] == "consolidated"
        assert await memory.maintenance.run_once(force=True) == 0
        assert len(memory.engine.store.get_all()) == 1
        assert memory.engine.store.get(target.id).version == 2
    finally:
        memory.engine.store.close()
