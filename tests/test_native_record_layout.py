"""Native readers retain v1 support and bind v2 content and fragment layout."""
from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.native_hosts.three_host_contract import _completed_batch_inspections, require_forget
from tests.test_native_batch_inspection import evidence
from tests.test_native_forget_review_pages import _calls
from tests.test_three_host_contract import TARGET


def upgrade(pages):
    fragments = [f for c in pages for f in c["result"]["fragments"]]
    encoded = json.dumps(
        {"format": "json_fragments_v2", "fragments": fragments},
        ensure_ascii=True, separators=(",", ":"), allow_nan=False,
    )
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    offset = 0
    for page in pages:
        page["args"]["cursor"] = None if not offset else f"{digest}:{offset}"
        review = page["result"]["review"]
        review.update(format="json_fragments_v2", layout_digest=digest)
        offset += len(page["result"]["fragments"])
        review["next_cursor"] = None if review["last_page"] else f"{digest}:{offset}"


@pytest.mark.parametrize("kind", ["batch", "preview"])
@pytest.mark.parametrize("defect", [None, "layout", "mixed_format", "legacy_cursor"])
def test_v2_layout_is_verified_in_addition_to_content(kind, defect):
    calls = evidence()[0] if kind == "batch" else _calls()
    pages = [c for c in calls if c["name"] == (
        "mnemos_inspect_batch" if kind == "batch" else "mnemos_forget_preview"
    )]
    upgrade(pages)
    original = deepcopy(calls)
    if defect == "layout":
        for page in pages:
            page["result"]["review"]["layout_digest"] = "f" * 64
            if page["args"].get("cursor"):
                page["args"]["cursor"] = "f" * 64 + ":" + page["args"]["cursor"].split(":")[1]
            review = page["result"]["review"]
            if review["next_cursor"]:
                review["next_cursor"] = "f" * 64 + ":" + review["next_cursor"].split(":")[1]
    elif defect == "mixed_format":
        pages[-1]["result"]["review"]["format"] = "json_fragments_v1"
    elif defect == "legacy_cursor":
        review = pages[1]["result"]["review"]
        pages[1]["args"]["cursor"] = review["snapshot_digest"] + ":1"
    def verify():
        if kind == "batch":
            return _completed_batch_inspections(calls)
        return require_forget(calls, TARGET)
    if defect:
        with pytest.raises(ValueError):
            verify()
    else:
        verify()
        assert calls == original
