"""Offline coverage for the bounded synthetic model forget diagnostic."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from copy import deepcopy

from benchmarks.native_hosts import model_forget_development_v1 as diagnostic
from mnemos.adapters.hermes_tools import dispatch_tool
from mnemos.client import MnemosClient


def test_cases_are_disjoint_and_options_are_fixed() -> None:
    markers = [case["old"] + case["new"] for case in diagnostic.CASES]
    assert len(set(markers)) == 3
    assert diagnostic.MODEL_OPTIONS == {
        "temperature": 0,
        "seed": 42,
        "presence_penalty": 0,
        "num_ctx": 32768,
        "num_predict": 2048,
        "top_k": 20,
        "top_p": 0.95,
    }
    assert {schema["name"] for schema in diagnostic.TOOLS} == {
        "mnemos_retrieve",
        "mnemos_inspect",
        "mnemos_remember",
        "mnemos_correct",
        "mnemos_forget",
        "mnemos_forget_preview",
        "mnemos_forget_related",
        "mnemos_job_status",
    }


def test_score_requires_review_and_persisted_erasure() -> None:
    seed = {
        "target_id": "target",
        "known_copy_ids": ["copy"],
        "known_source_event_ids": ["source"],
        "control_id": "control",
        "foreign_control_id": "foreign",
        "control": "control",
        "foreign": "foreign",
        "old": "old-marker",
        "new": "new-marker",
        "control_row": {"id": "control", "content": "control"},
        "foreign_control_row": {"id": "foreign", "content": "foreign"},
        "control_snapshot": {
            "fact": {"id": "control", "content": "control"},
            "evidence": [],
            "events": [],
            "revisions": [],
        },
        "foreign_control_snapshot": {
            "fact": {"id": "foreign", "content": "foreign"},
            "evidence": [],
            "events": [],
            "revisions": [],
        },
        "old_label": "old-marker",
        "new_label": "new-marker",
        "known_copy_rows": [
            {"id": "copy", "required_source_event_ids": ["source", "missing-source"]}
        ],
    }
    preview = {
        "status": "preview",
        "preview_digest": "a" * 64,
        "limit": 100,
        "complete": True,
    }
    calls = [
        {"operation": "tool_call", "name": "mnemos_retrieve", "args": {"query": "topic"}},
        {"operation": "tool_call", "name": "mnemos_inspect", "args": {"fact_id": "target"}},
        {
            "operation": "tool_call",
            "name": "mnemos_forget_preview",
            "args": {"fact_id": "target", "expected_version": 2},
        },
        {"operation": "tool_call", "name": "mnemos_inspect", "args": {"fact_id": "copy"}},
        {
            "operation": "tool_call",
            "name": "mnemos_forget_related",
            "args": {
                "fact_id": "target",
                "expected_version": 2,
                "preview_digest": "a" * 64,
                "preview_limit": 100,
                "selected_fact_ids": ["copy"],
                "selected_source_event_ids": ["source"],
            },
        },
        {"operation": "tool_result", "name": "mnemos_forget_preview", "result": preview},
        {
            "operation": "tool_result",
            "name": "mnemos_forget_related",
            "result": {
                "status": "committed",
                "forgotten": True,
                "selected_fact_ids": ["target", "copy"],
                "selected_source_event_ids": ["source"],
            },
        },
    ]
    state = {
        "facts": [{"id": "control", "content": "control"}, {"id": "foreign", "content": "foreign"}],
        "events": [],
        "evidence": [],
        "revisions": [],
        "jobs": [],
        "suppressions": [{"fact_id": "target"}, {"fact_id": "copy"}, {"source_event_id": "source"}],
    }
    gates = diagnostic._score(calls, seed, state)
    assert not gates["trace_correlated"]


def test_seed_uses_actual_local_service_and_two_scopes(tmp_path: Path) -> None:
    service, domain, context, foreign_context, _project, _foreign = diagnostic._service(tmp_path)
    try:
        seed = diagnostic._seed(service, domain, context, foreign_context, diagnostic.CASES[0])
        state = diagnostic._state(domain)
        assert seed["known_copy_ids"]
        assert seed["known_source_event_ids"]
        assert any(row["content"] == diagnostic.CASES[0]["control"] for row in state["facts"])
        assert any(row["content"] == diagnostic.CASES[0]["foreign"] for row in state["facts"])
        assert not set(seed["known_copy_ids"]) & {seed["target_id"]}
    finally:
        service.stop()
        domain.engine.store.close()


def test_actual_dispatch_positive_case_scores_all_gates(tmp_path: Path) -> None:
    async def run():
        service, domain, context, foreign_context, _project, _foreign = diagnostic._service(
            tmp_path
        )
        client = MnemosClient(descriptor_path=service.descriptor_path, auto_start=False)
        try:
            seed = diagnostic._seed(service, domain, context, foreign_context, diagnostic.CASES[0])
            records: list[dict[str, object]] = []
            call_number = 0

            async def call(name: str, args: dict[str, object]) -> dict[str, object]:
                nonlocal call_number
                call_id = f"response-0-call-{call_number}"
                call_number += 1
                records.append(
                    {
                        "operation": "model_response",
                        "tool_calls": [{"id": call_id, "name": name, "arguments": deepcopy(args)}],
                    }
                )
                result = json.loads(await dispatch_tool(client, context, name, args))
                records.append(
                    {
                        "operation": "tool_result",
                        "call_id": call_id,
                        "name": name,
                        "args": deepcopy(args),
                        "result": result,
                    }
                )
                return result

            await call("mnemos_retrieve", {"query": seed["new"], "top_k": 20})
            await call("mnemos_inspect", {"fact_id": seed["target_id"]})
            preview = await call(
                "mnemos_forget_preview",
                {"fact_id": seed["target_id"], "expected_version": 2},
            )
            selected = [
                item for item in preview["candidates"] if item["fact_id"] in seed["known_copy_ids"]
            ]
            for item in selected:
                await call(
                    "mnemos_inspect",
                    {"fact_id": item["fact_id"]},
                )
            source_ids = sorted(
                {source_id for item in selected for source_id in item["required_source_event_ids"]}
            )
            await call(
                "mnemos_forget_related",
                {
                    "fact_id": seed["target_id"],
                    "expected_version": 2,
                    "preview_digest": preview["preview_digest"],
                    "preview_limit": preview["limit"],
                    "selected_fact_ids": [item["fact_id"] for item in selected],
                    "selected_source_event_ids": source_ids,
                },
            )
            return records, seed, diagnostic._state(domain)
        finally:
            await client.close()
            service.stop()
            domain.engine.store.close()

    records, seed, state = asyncio.run(run())
    gates = diagnostic._score(records, seed, state)
    assert all(gates.values()), gates
    for change in (
        "missing_result",
        "wrong_call_id",
        "failed_inspection",
        "late_inspection",
        "missing_source_selection",
        "missing_source_suppression",
        "retained_job_label",
        "retained_revision_label",
        "control_version",
        "control_evidence",
        "extra_write",
    ):
        altered, final = deepcopy(records), deepcopy(state)
        if change == "missing_result":
            altered.pop()
        elif change == "wrong_call_id":
            altered[-1]["call_id"] = "unmatched"
        elif change == "failed_inspection":
            altered[3]["result"]["status"] = "rejected"
        elif change == "late_inspection":
            altered.extend(altered[6:8])
            del altered[6:8]
        elif change == "missing_source_selection":
            altered[-2]["tool_calls"][0]["arguments"]["selected_source_event_ids"] = []
            altered[-1]["args"]["selected_source_event_ids"] = []
        elif change == "missing_source_suppression":
            final["suppressions"] = [
                row
                for row in final["suppressions"]
                if row.get("source_event_id") != seed["known_source_event_ids"][0]
            ]
        elif change == "retained_job_label":
            final["jobs"].append(
                {
                    "id": "late",
                    "payload": json.dumps(
                        {"context": seed["context"], "content": seed["labels"][0]}
                    ),
                    "result": "{}",
                    "status": "pending",
                }
            )
        elif change == "retained_revision_label":
            final["revisions"].append(
                {"fact_id": seed["target_id"], "version": 1, "content": seed["labels"][0]}
            )
        elif change == "control_version":
            next(row for row in final["facts"] if row["id"] == seed["control_id"])["version"] += 1
        elif change == "control_evidence":
            final["evidence"] = [
                row for row in final["evidence"] if row["fact_id"] != seed["control_id"]
            ]
        elif change == "extra_write":
            args = {"content": "Unrequested new fact"}
            altered.extend(
                [
                    {
                        "operation": "model_response",
                        "tool_calls": [
                            {"id": "extra", "name": "mnemos_remember", "arguments": args}
                        ],
                    },
                    {
                        "operation": "tool_result",
                        "call_id": "extra",
                        "name": "mnemos_remember",
                        "args": args,
                        "result": {"schema_version": 1, "status": "committed"},
                    },
                ]
            )
        assert not all(diagnostic._score(altered, seed, final).values()), change
