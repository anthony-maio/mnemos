"""Context budgeting counts complete rendered records with an identified tokenizer."""

import pytest

from mnemos.context_budget import HostTokenizer, pack_context
from mnemos.retrieval import render_context_chunk
from mnemos.types import MemoryChunk


def chunk(identifier, content="Use Python", status="asserted"):
    return MemoryChunk(id=identifier, content=content, metadata={"assertion_status": status})


@pytest.mark.parametrize("status", ["asserted", "conflict", "unknown"])
def test_raw_capture_keeps_warning_regardless_of_assertion_label(status):
    record = chunk("raw", "A conversation claim.", status)
    record.metadata["source"] = "captured_passthrough"
    packed = pack_context([record], max_tokens=1000, top_k=1)
    assert "captured conversation; unverified, may be outdated" in packed.text
    assert ("unresolved conflict" in packed.text) == (status == "conflict")
    assert packed.token_count == len(packed.text.encode("utf-8"))
    assert record.metadata["assertion_status"] == status


def test_fallback_counts_utf8_including_identifiers_qualifiers_and_newlines():
    records = [chunk("a", "caf\u00e9", "conflict"), chunk("b", "\u6771\u4eac", "unknown")]
    expected = "".join(render_context_chunk(item) for item in records)
    packed = pack_context(records, max_tokens=len(expected.encode()), top_k=5)
    assert packed.text == expected
    assert packed.token_count == len(expected.encode())
    assert packed.token_count_method == "utf8_bytes_upper_bound"
    assert packed.tokenizer_name is None
    assert packed.fallback_reason is None
    assert pack_context(records, max_tokens=len(expected.encode()) - 1, top_k=5).chunks == (
        records[0],
    )


def test_host_counter_measures_whole_context_not_sum_of_record_costs():
    records = [chunk("a"), chunk("b")]
    calls = []

    def nonadditive(text):
        calls.append(text)
        return 9 if text.count("\n") == 2 else 4

    tokenizer = HostTokenizer("test-host", "revision-1", nonadditive)
    packed = pack_context(records, max_tokens=8, top_k=5, tokenizer=tokenizer)
    assert packed.chunks == (records[0],)
    assert "".join(render_context_chunk(item) for item in records) in calls
    assert packed.token_count == 4
    assert packed.token_count_method == "host_tokenizer"
    assert packed.tokenizer_name == "test-host"
    assert packed.tokenizer_revision == "revision-1"


def test_overlong_record_is_skipped_without_clipping_or_losing_later_small_fact():
    records = [chunk("long", "x" * 2000), chunk("small", "A whole fact.")]
    packed = pack_context(records, max_tokens=100, top_k=1)
    assert packed.chunks == (records[1],)
    assert packed.text == render_context_chunk(records[1])
    assert packed.omitted_ids == ("long",)


@pytest.mark.parametrize("bad_count", [True, -1, 0, 1.5, None])
def test_invalid_host_counts_fall_back_to_bytes(bad_count):
    record = chunk("a", "x" * 200)
    tokenizer = HostTokenizer("test-host", "r1", lambda text: bad_count)
    packed = pack_context([record], max_tokens=100, top_k=5, tokenizer=tokenizer)
    assert packed.chunks == ()
    assert packed.token_count == 0
    assert packed.token_count_method == "utf8_bytes_upper_bound"
    assert packed.fallback_reason == "invalid_token_count"


def test_counter_failure_restarts_packing_and_revisits_previously_skipped_records():
    records = [chunk("a"), chunk("b")]

    def fails_after_skip(text):
        if text.startswith("[a "):
            return 1000
        raise RuntimeError("tokenizer unavailable")

    packed = pack_context(
        records,
        max_tokens=100,
        top_k=5,
        tokenizer=HostTokenizer("test-host", "r1", fails_after_skip),
    )
    assert packed.chunks == tuple(records)
    assert packed.token_count == len(packed.text.encode())
    assert packed.token_count_method == "utf8_bytes_upper_bound"
    assert packed.fallback_reason == "tokenizer_error"
    assert packed.tokenizer_name == "test-host"


def test_empty_and_zero_budgets_do_not_call_tokenizer():
    def forbidden(text):
        raise AssertionError("empty context does not need tokenization")

    tokenizer = HostTokenizer("test-host", "r1", forbidden)
    for records, budget, top_k in [([], 10, 5), ([chunk("a")], 0, 5), ([chunk("a")], 10, 0)]:
        packed = pack_context(records, max_tokens=budget, top_k=top_k, tokenizer=tokenizer)
        assert packed.text == ""
        assert packed.token_count == 0
        assert packed.fallback_reason is None


@pytest.mark.parametrize("name,revision", [("", "r1"), ("host", ""), (" ", "r1")])
def test_tokenizer_identity_is_required(name, revision):
    with pytest.raises(ValueError):
        HostTokenizer(name, revision, len)
