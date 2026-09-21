"""SQLite regression coverage for support proposals racing with corrections."""

import asyncio
import threading
from datetime import datetime, timezone

import pytest

from mnemos.contracts import CaptureEvent, FactProposal
from tests.test_memory_service import service


def configure_sleep(memory):
    for name in type(memory.engine.config.modules).model_fields:
        setattr(memory.engine.config.modules, name, False)
    memory.engine.config.modules.sleep = True
    memory.engine.config.modules.affective = False


async def remember_target(memory, context, content="The release label is cedar-otter."):
    await memory.dispatch(
        "remember",
        {"context": context.model_dump(), "content": content, "event_id": "remember-target"},
    )
    assert await memory.maintenance.run_once(force=True) == 1
    target = memory.engine.store.get_all()[0]
    assert target.content == content
    assert target.version == 1
    return target


async def commit_correction(memory, context, target, content, owner):
    correction = await memory.dispatch(
        "correct",
        {
            "context": context.model_dump(),
            "fact_id": target.id,
            "expected_version": 1,
            "content": content,
            "event_id": f"{owner}-event",
        },
    )
    assert correction["processing_status"] == "pending"
    jobs = memory.engine.store.claim_jobs(owner=owner, limit=1, lease_seconds=300)
    assert len(jobs) == 1
    assert jobs[0].kind == "correct"
    await memory.process_job(jobs[0])


@pytest.mark.asyncio
async def test_support_cannot_commit_snapshot_after_target_correction(tmp_path):
    memory, context = service(tmp_path)
    old_content = "The release label is cedar-otter."
    new_content = "The release label is river-slate."
    support_event_id = "supporting-source"
    try:
        configure_sleep(memory)
        target = await remember_target(memory, context, old_content)

        async def support_proposal(events, _llm, *, existing_facts):
            assert any(fact.id == target.id and fact.version == 1 for fact in existing_facts)
            return [
                FactProposal(
                    content="Model wording must not replace the target content.",
                    evidence_ids=[events[0].id],
                    operation="support",
                    target_fact_id=target.id,
                    expected_version=1,
                )
            ]

        memory.engine.sleep_daemon.propose_facts = support_proposal

        captured = memory.capture(
            CaptureEvent(
                event_id=support_event_id,
                context=context,
                occurred_at=datetime.now(timezone.utc),
                role="user",
                content="Keep the release label convention in the project.",
                source_kind="codex_hook",
            )
        )
        assert captured["processing_status"] == "pending"
        support_event_id = captured["event_id"]

        entered_embedding = threading.Event()
        release_embedding = threading.Event()
        original_embed_document = memory.engine.embedder.embed_document

        def blocked_embed_document(text):
            if text == old_content and not release_embedding.is_set():
                entered_embedding.set()
                if not release_embedding.wait(timeout=10):
                    raise TimeoutError("support embedding was not released")
            return original_embed_document(text)

        memory.engine.embedder.embed_document = blocked_embed_document
        support_run = asyncio.create_task(memory.maintenance.run_once(force=True))
        assert await asyncio.to_thread(entered_embedding.wait, 10)

        await commit_correction(
            memory, context, target, new_content, owner="support-race-correction"
        )
        assert memory.engine.store.get(target.id).content == new_content
        assert memory.engine.store.get(target.id).version == 2

        release_embedding.set()
        await support_run

        current = memory.engine.store.get(target.id)
        assert current is not None
        assert current.content == new_content
        assert current.version == 2
        assert all(chunk.content != old_content for chunk in memory.engine.store.get_all())
        assert all(
            evidence.source_event_id != support_event_id
            for chunk in memory.engine.store.get_all()
            for evidence in memory.engine.store.list_fact_evidence(chunk.id)
        )
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_support_cannot_attach_evidence_after_model_snapshot_correction(tmp_path):
    memory, context = service(tmp_path)
    support_event_id = "support-before-read"
    old_content = "The export format is PDF."
    new_content = "The export format is HTML."
    model_returned = asyncio.Event()
    release_model = asyncio.Event()
    try:
        configure_sleep(memory)
        target = await remember_target(memory, context, old_content)

        async def support_proposal(events, _llm, *, existing_facts):
            assert any(fact.id == target.id and fact.version == 1 for fact in existing_facts)
            model_returned.set()
            await release_model.wait()
            return [
                FactProposal(
                    content="The export format is PDF.",
                    evidence_ids=[events[0].id],
                    operation="support",
                    target_fact_id=target.id,
                    expected_version=1,
                )
            ]

        memory.engine.sleep_daemon.propose_facts = support_proposal
        captured = memory.capture(
            CaptureEvent(
                event_id=support_event_id,
                context=context,
                occurred_at=datetime.now(timezone.utc),
                role="user",
                content="The export format remains the same.",
                source_kind="codex_hook",
            )
        )
        assert captured["processing_status"] == "pending"
        support_event_id = captured["event_id"]
        support_run = asyncio.create_task(memory.maintenance.run_once(force=True))
        await asyncio.wait_for(model_returned.wait(), timeout=10)

        await commit_correction(memory, context, target, new_content, owner="before-read-race")
        release_model.set()
        await support_run

        current = memory.engine.store.get(target.id)
        assert current is not None
        assert current.content == new_content
        assert current.version == 2
        assert all(
            evidence.source_event_id != support_event_id
            for evidence in memory.engine.store.list_fact_evidence(target.id)
        )
    finally:
        release_model.set()
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_support_success_attaches_evidence_without_revision_race(tmp_path):
    memory, context = service(tmp_path)
    support_event_id = "support-without-race"
    try:
        configure_sleep(memory)
        target = await remember_target(memory, context)

        async def support_proposal(events, _llm, *, existing_facts):
            assert any(fact.id == target.id and fact.version == 1 for fact in existing_facts)
            return [
                FactProposal(
                    content="Ignored model wording.",
                    evidence_ids=[events[0].id],
                    operation="support",
                    target_fact_id=target.id,
                    expected_version=1,
                )
            ]

        memory.engine.sleep_daemon.propose_facts = support_proposal
        captured = memory.capture(
            CaptureEvent(
                event_id=support_event_id,
                context=context,
                occurred_at=datetime.now(timezone.utc),
                role="user",
                content="The release label remains in use.",
                source_kind="codex_hook",
            )
        )
        assert captured["processing_status"] == "pending"
        support_event_id = captured["event_id"]
        assert await memory.maintenance.run_once(force=True) == 1

        current = memory.engine.store.get(target.id)
        assert current is not None
        assert current.content == target.content
        assert current.version == 1
        assert any(
            evidence.source_event_id == support_event_id
            for evidence in memory.engine.store.list_fact_evidence(target.id)
        )
        assert memory.engine.store.get_job(captured["job_id"]).status == "completed"
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("order", [("support", "correct"), ("correct", "support")])
async def test_support_and_correction_same_target_are_rejected_in_one_extraction(tmp_path, order):
    memory, context = service(tmp_path)
    source_event_id = f"mixed-{order[0]}-then-{order[1]}"
    old_content = "The deployment region is us-east-1."
    new_content = "The deployment region is eu-west-1."
    try:
        configure_sleep(memory)
        memory.engine.config.modules.mutable_rag = True
        target = await remember_target(memory, context, old_content)

        async def mixed_proposals(events, _llm, *, existing_facts):
            assert any(fact.id == target.id and fact.version == 1 for fact in existing_facts)
            proposals = {
                "support": FactProposal(
                    content="Ignored support wording.",
                    evidence_ids=[events[0].id],
                    operation="support",
                    target_fact_id=target.id,
                    expected_version=1,
                ),
                "correct": FactProposal(
                    content=new_content,
                    evidence_ids=[events[0].id],
                    operation="correct",
                    target_fact_id=target.id,
                    expected_version=1,
                ),
            }
            return [proposals[name] for name in order]

        memory.engine.sleep_daemon.propose_facts = mixed_proposals
        captured = memory.capture(
            CaptureEvent(
                event_id=source_event_id,
                context=context,
                occurred_at=datetime.now(timezone.utc),
                role="user",
                content="The deployment region changed.",
                source_kind="codex_hook",
            )
        )
        assert captured["processing_status"] == "pending"
        source_event_id = captured["event_id"]
        # Neither contradictory operation applies, in either order, and the
        # job finishes instead of exhausting the rest of the message.
        assert await memory.maintenance.run_once(force=True) == 1
        job = memory.engine.store.get_job(captured["job_id"])
        assert job is not None and job.status == "completed"
        assert [
            item["error"] for item in job.result["encoding_diagnostics"]["rejected_candidates"]
        ] == [
            "conflicting_target_operations",
            "conflicting_target_operations",
        ]

        current = memory.engine.store.get(target.id)
        assert current is not None
        assert current.content == old_content
        assert current.version == 1
        assert len(memory.engine.store.get_all()) == 1
        assert all(
            evidence.source_event_id != source_event_id
            for evidence in memory.engine.store.list_fact_evidence(target.id)
        )
    finally:
        memory.engine.store.close()
