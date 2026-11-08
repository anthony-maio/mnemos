from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.native_hosts.three_host_contract import (
    _completed_batch_inspections,
    _forget_inspected_before,
)
from benchmarks.native_hosts.three_host_contract import require_forget
from tests.test_three_host_contract import TARGET, _forget_calls


def evidence():
    results = [
        dict(
            schema_version=1,
            fact=dict(id=f"fact-{i}", version=2, content=f"Exact content {i}"),
            evidence=[dict(excerpt=f"source {i}")],
        )
        for i in range(3)
    ]
    snapshot = {"results": results}
    digest = hashlib.sha256(
        json.dumps(snapshot, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()
    calls = []
    for i, result in enumerate(results):
        calls.append(
            dict(
                name="mnemos_inspect_batch",
                args=dict(
                    fact_ids=[f"fact-{j}" for j in range(3)],
                    cursor=None if i == 0 else f"{digest}:{i}",
                ),
                result=dict(
                    schema_version=1,
                    status="inspection",
                    complete=i == 2,
                    review=dict(
                        format="json_fragments_v1",
                        snapshot_digest=digest,
                        first_fragment=i,
                        total_fragments=3,
                        last_page=i == 2,
                        next_cursor=None if i == 2 else f"{digest}:{i+1}",
                    ),
                    fragments=[dict(path=["results", i], value=result)],
                ),
            )
        )
    return calls, results


def test_every_selected_fact_can_be_proven_by_a_complete_batch():
    calls, results = evidence()
    assert _completed_batch_inspections(calls) == results
    for i in range(3):
        _forget_inspected_before(calls, 3, f"fact-{i}", 2, f"Exact content {i}")


@pytest.mark.parametrize(
    "mutation",
    [
        "first_missing",
        "middle_missing",
        "last_missing",
        "duplicate",
        "ids",
        "digest",
        "tamper",
        "cursor",
        "early_complete",
    ],
)
def test_incomplete_or_unbound_batch_never_counts_as_inspection(mutation):
    calls, _ = evidence()
    if mutation == "first_missing":
        calls.pop(0)
    elif mutation == "middle_missing":
        calls.pop(1)
    elif mutation == "last_missing":
        calls.pop()
    elif mutation == "duplicate":
        calls.insert(1, deepcopy(calls[0]))
    elif mutation == "ids":
        calls[1]["args"]["fact_ids"].reverse()
    elif mutation == "digest":
        calls[1]["result"]["review"]["snapshot_digest"] = "b" * 64
    elif mutation == "tamper":
        calls[1]["result"]["fragments"][0]["value"]["fact"]["content"] = "fabricated"
    elif mutation == "cursor":
        calls[1]["args"]["cursor"] = None
    else:
        calls[0]["result"]["complete"] = True
    with pytest.raises(ValueError):
        _forget_inspected_before(calls, len(calls), "fact-0", 2, "Exact content 0")


@pytest.mark.parametrize(
    "identifier,version,content",
    [
        ("absent", 2, "Exact content 0"),
        ("fact-0", 1, "Exact content 0"),
        ("fact-0", 2, "stale content"),
    ],
)
def test_selected_fact_identity_version_and_content_must_match(identifier, version, content):
    calls, _ = evidence()
    with pytest.raises(ValueError, match="prior inspection"):
        _forget_inspected_before(calls, len(calls), identifier, version, content)


def test_pages_after_mutation_cannot_complete_prior_inspection():
    calls, _ = evidence()
    with pytest.raises(ValueError, match="incomplete"):
        _forget_inspected_before(calls, 2, "fact-0", 2, "Exact content 0")


def batch_forget_calls():
    calls = _forget_calls()
    for index, call in enumerate(calls):
        if call["name"] != "mnemos_inspect":
            continue
        result = call["result"]
        snapshot = {"results": [result]}
        digest = hashlib.sha256(
            json.dumps(snapshot, ensure_ascii=True, separators=(",", ":")).encode()
        ).hexdigest()
        calls[index] = dict(
            name="mnemos_inspect_batch",
            model_call_correlated=True,
            args=dict(fact_ids=[result["fact"]["id"]]),
            result=dict(
                schema_version=1,
                status="inspection",
                complete=True,
                review=dict(
                    format="json_fragments_v1",
                    snapshot_digest=digest,
                    first_fragment=0,
                    total_fragments=1,
                    last_page=True,
                    next_cursor=None,
                ),
                fragments=[dict(path=[], value=snapshot)],
            ),
        )
    for index, call in enumerate(calls):
        call.update(
            invocation_line=index * 2 + 1, line=index * 2 + 2, response_id=f"response-{index}"
        )
    return calls


def test_full_forget_contract_accepts_complete_batches_but_not_missing_inspections():
    calls = batch_forget_calls()
    require_forget(calls, TARGET, require_batch=True)
    missing = [
        call
        for call in calls
        if not (call["name"] == "mnemos_inspect_batch" and call["args"]["fact_ids"] != [TARGET])
    ]
    with pytest.raises(ValueError, match="prior inspection"):
        require_forget(missing, TARGET)


def test_batch_protocol_cannot_pass_with_only_single_fact_inspections():
    calls = _forget_calls()
    for index, call in enumerate(calls):
        call.update(
            invocation_line=index * 2 + 1, line=index * 2 + 2, response_id=f"response-{index}"
        )
    require_forget(calls, TARGET)
    with pytest.raises(ValueError, match="requires complete"):
        require_forget(calls, TARGET, require_batch=True)


@pytest.mark.parametrize("overlap", ["none", "same_record", "split_record"])
def test_raw_claude_batch_correlation_requires_results_before_apply_invocation(tmp_path, overlap):
    from benchmarks.native_hosts.shared_trace import (
        TraceProvenance,
        load_source_records,
        parse_claude_tool_records,
    )

    calls = batch_forget_calls()
    rows = []
    for index, call in enumerate(calls):
        rows.append(
            dict(
                type="assistant",
                message=dict(
                    role="assistant",
                    id=f"response-{index}",
                    content=[
                        dict(
                            type="tool_use",
                            id=f"call-{index}",
                            name="mcp__mnemos__" + call["name"],
                            input=call["args"],
                        )
                    ],
                ),
            )
        )
        rows.append(
            dict(
                type="user",
                message=dict(
                    role="user",
                    content=[
                        dict(
                            type="tool_result",
                            tool_use_id=f"call-{index}",
                            content=json.dumps({"result": json.dumps(call["result"])}),
                        )
                    ],
                ),
            )
        )
    if overlap != "none":
        batch_index = max(
            i for i, call in enumerate(calls) if call["name"] == "mnemos_inspect_batch"
        )
        apply_index = next(
            i for i, call in enumerate(calls) if call["name"] == "mnemos_forget_related"
        )
        if overlap == "same_record":
            rows[batch_index * 2]["message"]["content"].extend(
                rows[apply_index * 2]["message"]["content"]
            )
            del rows[apply_index * 2]
        else:
            rows[apply_index * 2]["message"]["id"] = rows[batch_index * 2]["message"]["id"]
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    provenance = TraceProvenance(
        host="claude",
        process_id=1,
        session_id="test",
        source_file=str(path),
        source_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    parsed = parse_claude_tool_records(
        load_source_records(path, provenance=provenance),
        provenance=provenance,
        tool_prefix="mcp__mnemos__",
    )
    normalized = [
        dict(
            name=c.name,
            args=dict(c.arguments),
            result=dict(c.result),
            model_call_correlated=c.model_call_correlated,
            invocation_line=c.invocation_line_number,
            response_id=c.response_id,
            line=c.line_number,
        )
        for c in parsed
    ]
    assert all(c.invocation_line_sha256 for c in parsed)
    if overlap != "none":
        with pytest.raises(ValueError):
            require_forget(normalized, TARGET, require_batch=True)
    else:
        require_forget(normalized, TARGET, require_batch=True)
