"""Review typing must preserve records and participate in stale-preview checks."""

import pytest

from tests.test_related_forget import _preview_payload, _seed_overlap, _service


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_record", ["target", "candidate"])
@pytest.mark.parametrize("field", ["source", "assertion_status"])
async def test_review_types_capture_without_promoting_or_hiding_it(
    tmp_path, changed_record, field
):
    memory, context, _registry, _root = _service(tmp_path)
    store = memory.engine.store
    try:
        _assertion, target, captured = await _seed_overlap(memory, context)
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        candidate = next(c for c in preview["candidates"] if c["fact_id"] == captured.id)
        assert preview["target"]["record_kind"] == "memory"
        assert preview["target"]["assertion_status"] == "asserted"
        assert candidate["record_kind"] == "captured_evidence"
        assert candidate["assertion_status"] == "unknown"
        assert candidate["match_type"] == "exact_assertion"
        assert candidate["content"] == captured.content
        assert candidate["version"] == captured.version
        assert candidate["required_source_event_ids"]
        assert "not assertion status" in preview["matching_guidance"]

        # A metadata-only concurrent change must invalidate the reviewed type,
        # even if the content and fact version have not changed.
        changed = target if changed_record == "target" else captured
        value = (
            "conflict" if field == "assertion_status"
            else "captured_passthrough" if changed_record == "target"
            else "sleep_consolidation"
        )
        store._conn.execute(
            "UPDATE memory_chunks SET metadata=json_set(metadata, ?, ?) WHERE id=?",
            ("$." + field, value, changed.id),
        )
        store._conn.commit()
        assert store.get(changed.id).version == changed.version
        current = await memory.dispatch("forget_preview", _preview_payload(context, target))
        assert current["preview_digest"] != preview["preview_digest"]
        with pytest.raises(ValueError, match="stale"):
            await memory.dispatch(
                "forget_related",
                {
                    **_preview_payload(context, target),
                    "preview_digest": preview["preview_digest"],
                    "selected_fact_ids": [captured.id],
                    "selected_source_event_ids": candidate["required_source_event_ids"],
                },
            )
        assert store.get(target.id) is not None
        assert store.get(captured.id) is not None
    finally:
        store.close()
