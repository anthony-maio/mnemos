"""Native acceptance must bind every page and every short deletion selection."""

from copy import deepcopy
import json

import pytest

from benchmarks.native_hosts.three_host_contract import require_forget
from mnemos.forget_review import render_forget_review
from tests.test_three_host_contract import TARGET, _forget_calls, call


def _calls(*, paged=True, refs=True):
    original = _forget_calls()
    preview = original[2]["result"]
    preview["candidates"] = preview["candidates"][1:]
    preview["apply_preconditions"] = {
        key: original[3]["args"][key]
        for key in ["fact_id", "expected_version", "preview_digest", "preview_limit"]
    }
    preview["source_events"][0]["content"] = (
        ("Complete source text with \u96ea and a final qualification. " * 500)
        if paged
        else "Small source text."
    )
    pages = []
    cursor = None
    while True:
        args = {**original[2]["args"]}
        if cursor is not None:
            args["cursor"] = cursor
        result = json.loads(render_forget_review(preview, cursor))
        pages.append(call("mnemos_forget_preview", args, result))
        cursor = result.get("review", {}).get("next_cursor")
        if cursor is None:
            break
    write = original[-1]
    if refs:
        write["args"].pop("selected_fact_ids")
        write["args"].pop("selected_source_event_ids")
        write["args"].update(selected_fact_refs=["f1"], selected_source_refs=["s1"])
    return [*original[:2], *pages, write]


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("refs", [False, True])
def test_complete_review_and_exact_selected_receipt_pass_without_changing_trace(paged, refs):
    calls = _calls(paged=paged, refs=refs)
    original = deepcopy(calls)
    require_forget(calls, TARGET)
    assert calls == original


def _pages(calls):
    return [c for c in calls if c["name"] == "mnemos_forget_preview"]


@pytest.mark.parametrize(
    "defect",
    [
        "missing_first",
        "missing_last",
        "duplicate",
        "reorder",
        "wrong_cursor",
        "changed_snapshot",
        "changed_limit",
        "changed_count",
        "false_complete",
        "unknown_format",
        "early_preconditions",
        "text_gap",
        "text_overlap",
        "tampered_text",
        "missing_fragment",
        "boolean_path",
        "wrong_last_marker",
        "trailing_cursor",
        "wrong_final_preconditions",
        "float_page_version",
    ],
)
def test_incomplete_or_tampered_page_sequences_fail(defect):
    calls = _calls()
    pages = _pages(calls)
    assert len(pages) >= 3
    if defect == "missing_first":
        calls.remove(pages[0])
    elif defect == "missing_last":
        calls.remove(pages[-1])
    elif defect == "duplicate":
        calls.insert(3, deepcopy(pages[0]))
    elif defect == "reorder":
        calls[2], calls[3] = calls[3], calls[2]
    elif defect == "wrong_cursor":
        pages[1]["args"]["cursor"] = "f" * 64 + ":1"
    elif defect == "changed_snapshot":
        pages[1]["result"]["review"]["snapshot_digest"] = "b" * 64
    elif defect == "changed_limit":
        pages[1]["args"]["limit"] = 99
    elif defect == "changed_count":
        pages[1]["result"]["review"]["total_fragments"] += 1
    elif defect == "false_complete":
        pages[1]["result"]["complete"] = False
    elif defect == "unknown_format":
        pages[0]["result"]["review"]["format"] = "unverified_v2"
    elif defect == "early_preconditions":
        pages[0]["result"]["apply_preconditions"] = pages[-1]["result"]["apply_preconditions"]
    elif defect in {"text_gap", "text_overlap", "tampered_text"}:
        text = next(
            f for p in pages for f in p["result"]["fragments"] if "text" in f and f["offset"] > 0
        )
        if defect == "tampered_text":
            text["text"] = "tampered" + text["text"][8:]
        else:
            text["offset"] += 1 if defect == "text_gap" else -1
    elif defect == "missing_fragment":
        pages[1]["result"]["fragments"].pop(0)
    elif defect == "boolean_path":
        pages[0]["result"]["fragments"][0]["path"] = [True]
    elif defect == "wrong_last_marker":
        pages[0]["result"]["review"]["last_page"] = True
    elif defect == "trailing_cursor":
        pages[-1]["result"]["review"]["next_cursor"] = "a" * 64 + ":999"
    elif defect == "float_page_version":
        pages[1]["args"]["expected_version"] = 2.0
    else:
        pages[-1]["result"]["apply_preconditions"]["preview_limit"] = 99
    with pytest.raises(ValueError):
        require_forget(calls, TARGET)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize(
    "defect", ["missing_inspection", "missing_source", "unselected_dependency"]
)
def test_short_references_preserve_inspection_and_source_dependency_gates(paged, defect):
    calls = _calls(paged=paged)
    if defect == "missing_inspection":
        calls.pop(1)
    elif defect == "missing_source":
        calls[-1]["args"]["selected_source_refs"] = []
    else:
        # A receipt that deletes an extra fact cannot be accepted as reviewed.
        calls[-1]["result"]["selected_fact_ids"].append("unselected-dependent-fact")
    with pytest.raises(ValueError):
        require_forget(calls, TARGET)


@pytest.mark.parametrize(
    "defect", ["wrong_fact_ref", "wrong_source_ref", "duplicate", "mixed", "receipt", "blocked"]
)
def test_aliases_cannot_bypass_existing_selection_and_dependency_checks(defect):
    calls = _calls(paged=False)
    args = calls[-1]["args"]
    if defect == "wrong_fact_ref":
        args["selected_fact_refs"] = ["f2"]
    elif defect == "wrong_source_ref":
        args["selected_source_refs"] = ["s2"]
    elif defect == "duplicate":
        args["selected_fact_refs"] = ["f1", "f1"]
    elif defect == "mixed":
        args["selected_fact_ids"] = ["extra"]
    elif defect == "receipt":
        calls[-1]["result"]["selected_source_event_ids"] = []
    else:
        calls[2]["result"]["source_events"][0]["selection_allowed"] = False
    with pytest.raises(ValueError):
        require_forget(calls, TARGET)
