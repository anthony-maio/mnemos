"""Deletion receipts report bounded coverage without claiming semantic erasure."""

import json

import pytest

from mnemos.adapters.hermes_tools import dispatch_tool
from mnemos.client import MnemosClient
from mnemos.types import MemoryChunk
from tests.test_related_forget import _seed_overlap
from tests.test_related_forget_service_errors import _runtime


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["target_only", "reviewed", "direct"])
async def test_authenticated_receipt_distinguishes_target_only_and_reviewed_deletion(
    tmp_path, mode
):
    service, domain, context = _runtime(tmp_path)
    client = MnemosClient(descriptor_path=service.descriptor_path, auto_start=False)
    try:
        _, target, copy = await _seed_overlap(domain, context)
        # An unrelated but vector-similar fact is an advisory candidate, not a copy.
        advisory = MemoryChunk(
            content="The unrelated checksum remains stable.",
            embedding=list(target.embedding),
            metadata=dict(copy.metadata),
        )
        foreign = MemoryChunk(
            content=copy.content,
            embedding=list(target.embedding),
            metadata={**copy.metadata, "project_id": "foreign", "scope_id": "foreign"},
        )
        domain.engine.store.store(advisory)
        domain.engine.store.store(foreign)
        preview = json.loads(
            await dispatch_tool(
                client,
                context,
                "mnemos_forget_preview",
                {"fact_id": target.id, "expected_version": target.version},
            )
        )
        assert {row["fact_id"] for row in preview["candidates"]} == {copy.id, advisory.id}
        candidate = next(row for row in preview["candidates"] if row["fact_id"] == copy.id)
        args = {"fact_id": target.id, "expected_version": target.version}
        operation = "mnemos_forget" if mode == "direct" else "mnemos_forget_related"
        if mode != "direct":
            args.update(preview_digest=preview["preview_digest"], preview_limit=preview["limit"])
        if mode == "reviewed":
            args.update(
                selected_fact_ids=[copy.id],
                selected_source_event_ids=candidate["required_source_event_ids"],
            )
        result = json.loads(await dispatch_tool(client, context, operation, args))
        assert result["status"] == "committed"
        assert domain.engine.store.get(target.id) is None
        assert domain.engine.store.get(advisory.id) is not None
        assert domain.engine.store.get(foreign.id) is not None
        assert (domain.engine.store.get(copy.id) is None) == (mode == "reviewed")
        expected = {
            "selection": "reviewed_selection" if mode == "reviewed" else "target_only",
            "semantic_erasure_proven": False,
        }
        if mode != "direct":
            expected.update(
                retained_exact_fact_count=0 if mode == "reviewed" else 1,
                retained_advisory_fact_count=1,
                retained_eligible_source_event_count=0 if mode == "reviewed" else 1,
            )
        assert result["deletion_coverage"] == expected
    finally:
        await client.close()
        service.stop()
        domain.engine.store.close()


@pytest.mark.asyncio
async def test_removing_every_preview_record_does_not_prove_semantic_erasure(tmp_path):
    service, domain, context = _runtime(tmp_path)
    client = MnemosClient(descriptor_path=service.descriptor_path, auto_start=False)
    try:
        _, target, copy = await _seed_overlap(domain, context)
        payload = {
            "context": context.model_dump(mode="json"),
            "fact_id": target.id,
            "expected_version": target.version,
        }
        preview = await client.request("forget_preview", payload, queue_on_failure=False)
        result = await client.request(
            "forget_related",
            {
                **payload,
                "preview_digest": preview["preview_digest"],
                "preview_limit": preview["limit"],
                "selected_fact_ids": [copy.id],
                "selected_source_event_ids": [
                    r["source_event_id"] for r in preview["source_events"]
                ],
            },
            queue_on_failure=False,
        )
        assert result["deletion_coverage"] == {
            "selection": "reviewed_selection",
            "retained_exact_fact_count": 0,
            "retained_advisory_fact_count": 0,
            "retained_eligible_source_event_count": 0,
            "semantic_erasure_proven": False,
        }
        assert domain.engine.store.get_all() == []
    finally:
        await client.close()
        service.stop()
        domain.engine.store.close()
