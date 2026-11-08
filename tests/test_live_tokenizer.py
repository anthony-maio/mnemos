"""Live callback boundaries, including the packer's real fallback behavior."""

import json

import httpx
import pytest

from benchmarks.local_memory.live_tokenizer import LiveTokenCounter
from mnemos.context_budget import HostTokenizer, pack_context
from mnemos.types import MemoryChunk


def test_each_literal_count_makes_a_request_and_retains_raw_reply():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"tokens": [12, 34]})

    counter = LiveTokenCounter("http://127.0.0.1:43123", transport=httpx.MockTransport(handle))
    try:
        text = "caf\u00e9 <|endoftext|>"
        assert counter.count(text) == counter.count(text) == 2
        assert len(requests) == len(counter.records) == 2
        for request in requests:
            assert json.loads(request.content) == {
                "content": text,
                "add_special": False,
                "parse_special": False,
            }
        assert json.loads(counter.records[0]["response_text"]) == {"tokens": [12, 34]}
        assert counter.client.timeout.read == 2.0
        assert not counter.client.follow_redirects
    finally:
        counter.close()
    assert counter.client.is_closed


@pytest.mark.parametrize(
    "reply", [{"tokens": []}, {"tokens": [True]}, {"tokens": [-1]}, {"tokens": [1.2]}, [], {}]
)
def test_invalid_reply_restarts_whole_selection_with_byte_fallback(reply):
    calls = 0

    def handle(request):
        nonlocal calls
        calls += 1
        # Skip the first record using a valid native count, then fail on the
        # second. Byte fallback must revisit that first record.
        return httpx.Response(200, json={"tokens": [1] * 1000} if calls == 1 else reply)

    counter = LiveTokenCounter("http://127.0.0.1:43123", transport=httpx.MockTransport(handle))
    chunks = [MemoryChunk(id="a", content="A fact"), MemoryChunk(id="b", content="B fact")]
    try:
        ordinary = pack_context(chunks, max_tokens=200, top_k=5)
        result = pack_context(
            chunks,
            max_tokens=200,
            top_k=5,
            tokenizer=HostTokenizer("native", "test", counter.count),
        )
        assert result.text == ordinary.text
        assert result.chunks == ordinary.chunks
        assert result.token_count == ordinary.token_count
        assert result.token_count_method == "utf8_bytes_upper_bound"
        assert result.fallback_reason == "tokenizer_error"
        assert counter.records[-1]["error_type"] == "ValueError"
    finally:
        counter.close()


def test_timeout_is_recorded_and_reaches_fallback():
    def handle(request):
        raise httpx.ReadTimeout("bounded test failure", request=request)

    counter = LiveTokenCounter("http://127.0.0.1:43123", transport=httpx.MockTransport(handle))
    try:
        result = pack_context(
            [MemoryChunk(id="a", content="Whole fact")],
            max_tokens=200,
            top_k=1,
            tokenizer=HostTokenizer("native", "test", counter.count),
        )
        assert len(result.chunks) == 1
        assert result.fallback_reason == "tokenizer_error"
        assert counter.records[0]["error_type"] == "ReadTimeout"
    finally:
        counter.close()


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:80",
        "http://example.com:80",
        "http://localhost:80",
        "http://user@127.0.0.1:80",
        "http://127.0.0.1:80/path",
        "http://127.0.0.1:80?x=1",
        "http://127.0.0.1",
    ],
)
def test_nonliteral_or_ambiguous_origin_is_rejected(endpoint):
    with pytest.raises(ValueError):
        LiveTokenCounter(endpoint)
