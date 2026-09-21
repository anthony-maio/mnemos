import json
from datetime import datetime, timezone

import pytest

from mnemos.contracts import CaptureEvent
from mnemos.types import MemoryChunk
from mnemos.utils.llm import LLMProvider
from mnemos.utils.storage import compute_canonical_key
from tests.test_memory_service import service


class ExactDeletedAssertion(LLMProvider):
    async def predict(self, prompt, **kwargs):
        event = json.loads(prompt.split("SOURCE EVENTS:\n", 1)[1])[0]
        return json.dumps(
            {
                "decision": "facts",
                "facts": [
                    {
                        "operation": "add",
                        "content": "The release label is cedar-otter.",
                        "evidence_ids": [event["id"]],
                    }
                ],
            }
        )

    async def classify(self, prompt, labels):
        return {}


class ExactAndUnrelatedAssertions(ExactDeletedAssertion):
    async def predict(self, prompt, **kwargs):
        event = json.loads(prompt.split("SOURCE EVENTS:\n", 1)[1])[0]
        return json.dumps(
            {
                "decision": "facts",
                "facts": [
                    {
                        "operation": "add",
                        "content": "The release label is cedar-otter.",
                        "evidence_ids": [event["id"]],
                    },
                    {
                        "operation": "add",
                        "content": "Keep river-slate in release notes.",
                        "evidence_ids": [event["id"]],
                    },
                ],
            }
        )


class DifferentlyWordedSupport(LLMProvider):
    async def predict(self, prompt, **kwargs):
        source = json.loads(prompt.split("SOURCE EVENTS:\n", 1)[1].split("\nEXISTING FACTS", 1)[0])[
            0
        ]
        facts = json.loads(
            prompt.split("EXISTING FACTS (data, not instructions):\n", 1)[1].split(
                "\nFor a genuinely", 1
            )[0]
        )
        return json.dumps(
            {
                "decision": "facts",
                "facts": [
                    {
                        "operation": "support",
                        "content": "A differently worded release convention.",
                        "target_fact_id": facts[0]["id"],
                        "evidence_ids": [source["id"]],
                    }
                ],
            }
        )

    async def classify(self, prompt, labels):
        return {}


@pytest.mark.asyncio
async def test_assistant_confirmation_cannot_recreate_forgotten_assertion(tmp_path):
    memory, context = service(tmp_path)
    for name in type(memory.engine.config.modules).model_fields:
        setattr(memory.engine.config.modules, name, False)
    memory.engine.config.modules.sleep = True
    memory.engine.config.modules.affective = False
    memory.engine._llm = ExactDeletedAssertion()
    try:
        await memory.dispatch(
            "remember",
            {"context": context.model_dump(), "content": "The release label is cedar-otter."},
        )
        await memory.maintenance.run_once(force=True)
        target = memory.engine.store.get_all()[0]
        captured = memory.capture(
            CaptureEvent(
                event_id="assistant-confirmation",
                context=context,
                occurred_at=datetime.now(timezone.utc),
                role="assistant",
                content="The release label is cedar-otter.",
                source_kind="codex_hook",
            )
        )
        assert captured["status"] == "committed"
        assert captured["processing_status"] == "pending"
        await memory.dispatch(
            "forget",
            {"context": context.model_dump(), "fact_id": target.id, "expected_version": 1},
        )
        await memory.maintenance.run_once(force=True)
        assert not any("cedar-otter" in chunk.content for chunk in memory.engine.store.get_all())
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_atomic_extract_commit_checks_unprefixed_suppression(tmp_path):
    memory, context = service(tmp_path)
    try:
        captured = memory.capture(
            CaptureEvent(
                event_id="pending-assistant-extraction",
                context=context,
                occurred_at=datetime.now(timezone.utc),
                role="assistant",
                content="The assistant confirmed deletion.",
                source_kind="codex_hook",
            )
        )
        assert captured["status"] == "committed"
        assert captured["processing_status"] == "pending"
        job = memory.engine.store.claim_jobs(owner="test-owner", limit=1)[0]
        canonical = compute_canonical_key(
            "The release label is cedar-otter.",
            "project",
            context.project_id,
            context.user_profile_id,
        )
        memory.engine.store.record_suppression(canonical_key=canonical, reason="explicit_forget")
        candidate = MemoryChunk(
            content="Assistant reported: The release label is cedar-otter.",
            metadata={
                "scope": "project",
                "scope_id": context.project_id,
                "user_profile_id": context.user_profile_id,
                "source_roles": ["user"],
                "source_event_ids": [job.payload["event_id"]],
            },
            canonical_key=compute_canonical_key(
                "Assistant reported: The release label is cedar-otter.",
                "project",
                context.project_id,
                context.user_profile_id,
            ),
        )
        committed = memory.engine.store.commit_fact_and_job(
            job.id,
            owner="test-owner",
            lease_token=job.lease_token,
            chunks=[candidate],
            processed_event_ids=[job.payload["event_id"]],
            result={"decision": "facts"},
        )
        assert committed
        assert memory.engine.store.get_all() == []
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("candidate_scope", "candidate_profile"),
    [
        ("other-project", "local"),
        ("same-project", "other-profile"),
    ],
    ids=["other-project-same-profile", "same-project-other-profile"],
)
async def test_raw_suppression_isolated_by_project_and_profile(
    tmp_path, candidate_scope, candidate_profile
):
    memory, context = service(tmp_path)
    try:
        captured = memory.capture(
            CaptureEvent(
                event_id="cross-scope-extraction",
                context=context,
                occurred_at=datetime.now(timezone.utc),
                role="assistant",
                content="The assistant confirmed deletion.",
                source_kind="codex_hook",
            )
        )
        job = memory.engine.store.claim_jobs(owner="scope-test", limit=1)[0]
        source_canonical = compute_canonical_key(
            "The release label is cedar-otter.",
            "project",
            context.project_id,
            context.user_profile_id,
        )
        memory.engine.store.record_suppression(
            canonical_key=source_canonical, reason="explicit_forget"
        )
        if candidate_scope == "same-project":
            candidate_scope = context.project_id
        candidate = MemoryChunk(
            content="Assistant reported: The release label is cedar-otter.",
            metadata={
                "scope": "project",
                "scope_id": candidate_scope,
                "user_profile_id": candidate_profile,
                "source_roles": ["user"],
                "source_event_ids": [job.payload["event_id"]],
            },
        )
        assert captured["processing_status"] == "pending"
        assert memory.engine.store.commit_fact_and_job(
            job.id,
            owner="scope-test",
            lease_token=job.lease_token,
            chunks=[candidate],
            processed_event_ids=[job.payload["event_id"]],
            result={"decision": "facts"},
        )
        stored = memory.engine.store.get(candidate.id)
        assert stored is not None
        assert stored.metadata["scope_id"] == candidate_scope
        assert stored.metadata["user_profile_id"] == candidate_profile
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_unrelated_extraction_survives_suppressed_proposal(tmp_path):
    memory, context = service(tmp_path)
    for name in type(memory.engine.config.modules).model_fields:
        setattr(memory.engine.config.modules, name, False)
    memory.engine.config.modules.sleep = True
    memory.engine.config.modules.affective = False
    memory.engine._llm = ExactAndUnrelatedAssertions()
    try:
        await memory.dispatch(
            "remember",
            {"context": context.model_dump(), "content": "The release label is cedar-otter."},
        )
        await memory.maintenance.run_once(force=True)
        target = memory.engine.store.get_all()[0]
        captured = memory.capture(
            CaptureEvent(
                event_id="mixed-extraction",
                context=context,
                occurred_at=datetime.now(timezone.utc),
                role="assistant",
                # A bare copy of the forgotten text is removed by the forget
                # itself; a mixed message reaches per-proposal suppression.
                content="The release label is cedar-otter. Keep river-slate in release notes.",
                source_kind="codex_hook",
            )
        )
        assert captured["processing_status"] == "pending"
        await memory.dispatch(
            "forget",
            {"context": context.model_dump(), "fact_id": target.id, "expected_version": 1},
        )
        await memory.maintenance.run_once(force=True)
        contents = [chunk.content for chunk in memory.engine.store.get_all()]
        assert "Assistant reported: Keep river-slate in release notes." in contents
        assert not any("cedar-otter" in content for content in contents)
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_paraphrased_assistant_capture_remains_unknown(tmp_path):
    memory, context = service(tmp_path)
    for name in type(memory.engine.config.modules).model_fields:
        setattr(memory.engine.config.modules, name, False)
    try:
        await memory.dispatch(
            "remember",
            {"context": context.model_dump(), "content": "The release label is cedar-otter."},
        )
        await memory.maintenance.run_once(force=True)
        target = memory.engine.store.get_all()[0]
        captured = memory.capture(
            CaptureEvent(
                event_id="paraphrased-confirmation",
                context=context,
                occurred_at=datetime.now(timezone.utc),
                role="assistant",
                content="I removed the old release-label convention as requested.",
                source_kind="codex_hook",
            )
        )
        assert captured["processing_status"] == "pending"
        await memory.dispatch(
            "forget",
            {"context": context.model_dump(), "fact_id": target.id, "expected_version": 1},
        )
        await memory.maintenance.run_once(force=True)
        copies = [
            chunk
            for chunk in memory.engine.store.get_all()
            if "removed the old release-label" in chunk.content
        ]
        assert len(copies) == 1
        assert copies[0].metadata["assertion_status"] == "unknown"
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_explicit_remember_can_reteach_after_forget(tmp_path):
    memory, context = service(tmp_path)
    try:
        first = await memory.dispatch(
            "remember",
            {"context": context.model_dump(), "content": "The release label is cedar-otter."},
        )
        await memory.maintenance.run_once(force=True)
        target = memory.engine.store.get_all()[0]
        await memory.dispatch(
            "forget",
            {"context": context.model_dump(), "fact_id": target.id, "expected_version": 1},
        )
        taught = await memory.dispatch(
            "remember",
            {
                "context": context.model_dump(),
                "event_id": "explicit-reteach",
                "content": "The release label is cedar-otter.",
            },
        )
        assert taught["status"] == "committed"
        await memory.maintenance.run_once(force=True)
        assert any(
            chunk.content == "The release label is cedar-otter."
            for chunk in memory.engine.store.get_all()
        )
        assert first["event_id"] != taught["event_id"]
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_support_preserves_target_content_and_adds_evidence(tmp_path):
    memory, context = service(tmp_path)
    for name in type(memory.engine.config.modules).model_fields:
        setattr(memory.engine.config.modules, name, False)
    memory.engine.config.modules.sleep = True
    memory.engine.config.modules.affective = False
    memory.engine._llm = DifferentlyWordedSupport()
    try:
        await memory.dispatch(
            "remember",
            {"context": context.model_dump(), "content": "The release label is cedar-otter."},
        )
        await memory.maintenance.run_once(force=True)
        target = memory.engine.store.get_all()[0]
        captured = memory.capture(
            CaptureEvent(
                event_id="supporting-source",
                context=context,
                occurred_at=datetime.now(timezone.utc),
                role="user",
                content="Keep the release label convention in the project.",
                source_kind="codex_hook",
            )
        )
        assert captured["processing_status"] == "pending"
        await memory.maintenance.run_once(force=True)
        updated = memory.engine.store.get(target.id)
        assert updated is not None
        assert updated.content == "The release label is cedar-otter."
        assert updated.version == target.version
        assert len(memory.engine.store.get_all()) == 1
        evidence = memory.engine.store.list_fact_evidence(target.id)
        assert any(item.source_event_id == captured["event_id"] for item in evidence)
    finally:
        memory.engine.store.close()
