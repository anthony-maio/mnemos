"""Scope denial is a structured service outcome, not necessarily an MCP error."""

import json
from pathlib import Path
import runpy

import pytest

CHECK = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "benchmarks/native_hosts/registry_mcp_session.py")
)["require_scope_denial"]


def denial():
    return dict(
        schema_version=1,
        status="degraded",
        committed=False,
        degraded=True,
        error="Mnemos service returned HTTP 503 (scope_unavailable)",
    )


def test_structured_scope_denial_is_recognized():
    CHECK(json.dumps(denial()))


def test_store_denial_requires_exact_scope_and_no_committed_event():
    value = dict(denial(), event_committed=False, scope="project", scope_id="expected")
    CHECK(json.dumps(value), store_scope_id="expected")
    with pytest.raises(AssertionError):
        CHECK(json.dumps(value), store_scope_id="wrong")
    value["event_committed"] = True
    with pytest.raises(AssertionError):
        CHECK(json.dumps(value), store_scope_id="expected")


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "committed"},
        {"committed": True},
        {"degraded": False},
        {"error": "Provider unavailable"},
        {"facts": [{"content": "private"}]},
        {"schema_version": 2},
    ],
)
def test_other_errors_or_leaked_content_cannot_count_as_scope_denial(changes):
    value = denial()
    value.update(changes)
    with pytest.raises(AssertionError):
        CHECK(json.dumps(value))
