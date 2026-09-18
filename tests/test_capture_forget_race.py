from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import asyncio
import hashlib
import json
import threading

import pytest

from mnemos.contracts import CaptureEvent
from mnemos.claude_identity import legacy_event_id, scoped_event_id
from mnemos.utils.storage import SQLiteStore
from tests.test_memory_service import service


def assert_forgotten_after_reopen(tmp_path):
    reopened = SQLiteStore(str(tmp_path / "memory.sqlite"))
    try:
        assert reopened.get_all() == []
        assert reopened.list_episodic_events() == []
        assert reopened.list_jobs(status="pending") == []
    finally:
        reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["user", "assistant"])
async def test_forget_committed_before_capture_cannot_restore_raw_source(
    tmp_path, monkeypatch, role
):
    memory, context = service(tmp_path)
    entered, proceed = threading.Event(), threading.Event()
    store = memory.engine.store
    try:
        for flag in type(memory.engine.config.modules).model_fields:
            setattr(memory.engine.config.modules, flag, False)
        await memory.dispatch(
            "remember",
            {"context": context.model_dump(), "content": "Use cedar-otter release labels."},
        )
        await memory.maintenance.run_once(force=True)
        target = store.get_all()[0]
        capture = store.capture_event_and_job

        def delayed_capture(event, job, **kwargs):
            entered.set()
            assert proceed.wait(5), "Forget did not release the capture barrier"
            return capture(event, job, **kwargs)

        monkeypatch.setattr(store, "capture_event_and_job", delayed_capture)
        event = CaptureEvent(
            event_id="late-automatic-capture",
            context=context,
            occurred_at=datetime.now(timezone.utc),
            role=role,
            content=target.content,
            source_kind="codex_hook",
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(memory.capture, event)
            try:
                assert entered.wait(5), "Capture did not reach its persistence boundary"
                # A separate SQLite connection commits the actual forget while
                # capture has already passed its service-level suppression check.
                other = SQLiteStore(str(tmp_path / "memory.sqlite"))
                try:
                    assert other.forget_fact(target.id, expected_version=target.version)
                finally:
                    other.close()
            finally:
                proceed.set()
            response = pending.result(timeout=5)

        assert store.list_episodic_events() == []
        assert store.get_job("event:" + response["event_id"]) is None
        assert response["processing_status"] == "suppressed"
        assert response["job_id"] is None
        assert store.get_all() == []
        assert await memory.maintenance.run_once(force=True) == 0
    finally:
        proceed.set()
        store.close()

    assert_forgotten_after_reopen(tmp_path)


@pytest.mark.asyncio
async def test_late_correction_cannot_reintroduce_source_for_forgotten_target(
    tmp_path, monkeypatch
):
    memory, context = service(tmp_path)
    entered, proceed = threading.Event(), threading.Event()
    store = memory.engine.store
    try:
        await memory.dispatch(
            "remember", {"context": context.model_dump(), "content": "Use cedar-otter labels."}
        )
        await memory.maintenance.run_once(force=True)
        target = store.get_all()[0]
        capture = store.capture_event_and_job

        def delayed_capture(event, job, **kwargs):
            entered.set()
            assert proceed.wait(5)
            return capture(event, job, **kwargs)

        monkeypatch.setattr(store, "capture_event_and_job", delayed_capture)
        payload = {
            "context": context.model_dump(),
            "fact_id": target.id,
            "expected_version": target.version,
            "content": "Use maple-heron labels instead of cedar-otter.",
        }
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(asyncio.run, memory.dispatch("correct", payload))
            try:
                assert entered.wait(5)
                await memory.dispatch("forget", payload)
            finally:
                proceed.set()
            response = pending.result(timeout=5)
        assert store.list_episodic_events() == []
        assert response["processing_status"] == "suppressed"
        assert response["job_id"] is None
        assert store.get_job("event:" + response["event_id"]) is None
    finally:
        proceed.set()
        store.close()

    assert_forgotten_after_reopen(tmp_path)


def test_verified_claude_alias_is_rechecked_at_capture_commit(tmp_path, monkeypatch):
    memory, context = service(tmp_path)
    context = context.model_copy(update={"host": "claude-code"})
    content, source_ref = "Keep river-slate release notes.", "native-message-one"
    event = CaptureEvent(
        event_id=scoped_event_id(context, source_ref, "user", content),
        context=context,
        occurred_at=datetime.now(timezone.utc),
        role="user",
        content=content,
        source_kind="claude_code_transcript",
        source_ref=source_ref,
        identity_version=2,
    )
    old_id = legacy_event_id(context.session_id, source_ref, "user", content)
    old_key = hashlib.sha256(
        json.dumps([context.user_profile_id, context.host, context.session_id, old_id]).encode(
            "utf-8"
        )
    ).hexdigest()
    store = memory.engine.store
    capture = store.capture_event_and_job

    def suppress_before_commit(event, job, **kwargs):
        other = SQLiteStore(str(tmp_path / "memory.sqlite"))
        try:
            other.record_suppression(source_event_id=old_key, reason="explicit_forget")
        finally:
            other.close()
        return capture(event, job, **kwargs)

    monkeypatch.setattr(store, "capture_event_and_job", suppress_before_commit)
    try:
        response = memory.capture(event)
        assert response["processing_status"] == "suppressed"
        assert response["job_id"] is None
        assert store.list_episodic_events() == []
        assert store.list_jobs() == []
    finally:
        store.close()

    assert_forgotten_after_reopen(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_pending_replay_cannot_restore_source_forgotten_during_capture(
    tmp_path, monkeypatch, explicit
):
    memory, context = service(tmp_path)
    entered, proceed = threading.Event(), threading.Event()
    store = memory.engine.store
    try:
        for flag in type(memory.engine.config.modules).model_fields:
            setattr(memory.engine.config.modules, flag, False)
        event = CaptureEvent(
            event_id="original-source",
            context=context,
            occurred_at=datetime.now(timezone.utc),
            role="user",
            content="Use silver-birch deployment labels.",
            source_kind="explicit_remember" if explicit else "codex_hook",
        )
        original = memory.capture(event, explicit=explicit)
        capture = store.capture_event_and_job

        def delayed_capture(event, job, **kwargs):
            entered.set()
            assert proceed.wait(5), "Forget did not release the replay barrier"
            return capture(event, job, **kwargs)

        monkeypatch.setattr(store, "capture_event_and_job", delayed_capture)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(memory.capture, event, explicit=explicit)
            try:
                assert entered.wait(5), "Replay did not reach its persistence boundary"
                assert await memory.maintenance.run_once(force=True) == 1
                target = store.get_all()[0]
                await memory.dispatch(
                    "forget",
                    {
                        "context": context.model_dump(),
                        "fact_id": target.id,
                        "expected_version": target.version,
                    },
                )
            finally:
                proceed.set()
            response = pending.result(timeout=5)
        assert store.list_episodic_events() == []
        assert response["event_id"] == original["event_id"]
        assert response["processing_status"] == "suppressed"
        assert response["job_id"] is None
        job = store.get_job(original["job_id"])
        assert job.status == "completed"
        assert job.result == {"decision": "forgotten"}
        assert store.get_all() == []
    finally:
        proceed.set()
        store.close()

    assert_forgotten_after_reopen(tmp_path)
