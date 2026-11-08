import asyncio
from copy import deepcopy
import json

import pytest

from mnemos.inspection_batch import render_inspection_batch
from benchmarks.native_hosts.three_host_contract import _review_snapshot
from tests.test_forget_review_paging import _tools


def inspection(identifier="fact-a", version=1):
    return {
        "schema_version": 1,
        "fact": {
            "id": identifier,
            "version": version,
            "content": 'Full source "quoted" \u2603 ' * 1500,
        },
        "evidence": [{"excerpt": "exact source " * 1500}],
        "revisions": [{"version": 1, "content": "old content"}],
    }


def test_pages_preserve_every_field_and_stay_within_wire_bound():
    results = [inspection(), inspection("fact-b", 2)]
    frozen = deepcopy(results)
    fragments, cursor, pages = [], None, 0
    while True:
        text = render_inspection_batch(results, cursor)
        assert len(json.dumps({"result": text}).encode()) <= 12000
        page = json.loads(text)
        assert page["review"]["first_fragment"] == len(fragments)
        fragments.extend(page["fragments"])
        pages += 1
        cursor = page["review"]["next_cursor"]
        if cursor is None:
            assert page["complete"] and page["review"]["last_page"]
            break
        assert not page["complete"]
    assert pages > 1
    assert _review_snapshot(fragments) == {"results": frozen}
    assert results == frozen


@pytest.mark.parametrize("change", ["version", "evidence", "order"])
def test_changes_in_any_part_invalidate_cursor(change):
    results = [inspection(), inspection("fact-b")]
    cursor = json.loads(render_inspection_batch(results))["review"]["next_cursor"]
    assert cursor
    if change == "version":
        results[0]["fact"]["version"] += 1
    elif change == "evidence":
        results[1]["evidence"][0]["excerpt"] += " changed"
    else:
        results.reverse()
    with pytest.raises(ValueError, match="changed"):
        render_inspection_batch(results, cursor)


@pytest.mark.parametrize(
    "results",
    [[], [inspection()] * 21, [inspection(), inspection()], [{"schema_version": 1, "fact": None}]],
)
def test_incomplete_duplicate_and_oversized_batches_rejected(results):
    with pytest.raises(ValueError):
        render_inspection_batch(results)


def test_every_requested_id_uses_same_resolved_scope_and_order():
    class Client:
        def __init__(self):
            self.calls = []

        async def request(self, operation, payload, **kwargs):
            self.calls.append((operation, deepcopy(payload)))
            return inspection(payload["fact_id"])

    client = Client()
    tools, ctx = _tools(client)
    page = json.loads(asyncio.run(tools["mnemos_inspect_batch"](["fact-a", "fact-b"], ctx=ctx)))
    assert page["status"] == "inspection"
    assert [payload["fact_id"] for _, payload in client.calls] == ["fact-a", "fact-b"]
    assert all(
        op == "inspect"
        and p["context"]["project_id"] == "project-a"
        and p["context"]["user_profile_id"] == "profile-a"
        for op, p in client.calls
    )


def test_unavailable_fact_prevents_partial_results_from_escaping():
    class Client:
        async def request(self, operation, payload, **kwargs):
            return (
                inspection()
                if payload["fact_id"] == "fact-a"
                else {"schema_version": 1, "fact": None}
            )

    tools, ctx = _tools(Client())
    with pytest.raises(ValueError, match="available"):
        asyncio.run(tools["mnemos_inspect_batch"](["fact-a", "foreign"], ctx=ctx))
