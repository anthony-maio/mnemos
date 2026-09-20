"""Independently check raw embedding responses against the exported vector map."""
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
payload = json.loads((root / "artifact.json").read_text(encoding="utf-8"))
assert payload["status"] == "complete"
assert payload["git_revision_before"] == payload["git_revision_after"]
assert payload["endpoint"] == "http://127.0.0.1:11434"
assert payload["embedding_text_contract"] == {
    "query_prefix": "", "document_prefix": "", "symmetric_prefix": "", "symmetry": "symmetric"
}
manifest_path = Path(payload["manifest"]["path"])
manifest_bytes = manifest_path.read_bytes()
assert hashlib.sha256(manifest_bytes).hexdigest() == payload["manifest"]["sha256"]
manifest = json.loads(manifest_bytes)
texts = []
for case in manifest["graph_cases"]:
    texts.append(case["query"])
    for key in ("facts", "events", "scope_controls"):
        texts.extend(row["content"] for row in case.get(key, []))
    for action in case.get("lifecycle_actions", []):
        if "new_content" in action:
            texts.append(action["new_content"])
unique = list(dict.fromkeys(texts))
observed = []
embed_requests = 0
for expected_index, row in enumerate(payload["requests"], 1):
    assert row["index"] == expected_index
    assert row["url"].startswith(payload["endpoint"] + "/api/")
    assert 200 <= row["response_status"] < 300
    stem = f"request-{expected_index:04d}"
    response_bytes = (root / "raw" / (stem + ".response.bin")).read_bytes()
    assert len(response_bytes) == row["response_bytes"]
    assert hashlib.sha256(response_bytes).hexdigest() == row["response_sha256"]
    request_path = root / "raw" / (stem + ".request.bin")
    request_bytes = request_path.read_bytes() if request_path.exists() else b""
    assert len(request_bytes) == row["request_bytes"]
    assert (hashlib.sha256(request_bytes).hexdigest() if request_bytes else None) == row["request_sha256"]
    if row["url"].endswith("/api/embed"):
        embed_requests += 1
        request = json.loads(request_bytes)
        response = json.loads(response_bytes)
        assert request["model"] == payload["model"]
        assert response["model"] == payload["model"]
        assert request["truncate"] is False
        assert 1 <= len(request["input"]) <= 16
        assert len(request["input"]) == len(response["embeddings"])
        for text, vector in zip(request["input"], response["embeddings"]):
            assert len(vector) == 768
            assert all(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) for value in vector)
            assert any(value != 0 for value in vector)
            key = hashlib.sha256(text.encode()).hexdigest()
            assert payload["vectors"][key] == {"text": text, "vector": vector}
            observed.append(text)
assert observed == unique
assert len(payload["vectors"]) == len(unique)
assert len(payload["bindings"]) == len(texts)
assert [item["text"] for item in payload["bindings"]] == texts
assert all(item["text_sha256"] == hashlib.sha256(item["text"].encode()).hexdigest() for item in payload["bindings"])
for position in ("before", "after"):
    matches = [item for item in payload["tags_" + position]["models"] if item["name"] == payload["model"]]
    assert len(matches) == 1 and matches[0]["digest"] == payload["expected_digest"]
show = lambda value: {key: item for key, item in value.items() if key != "modified_at"}
assert show(payload["show_before"]) == show(payload["show_after"])
print(json.dumps({"status": "verified", "artifact_sha256": hashlib.sha256((root / "artifact.json").read_bytes()).hexdigest(), "source_commit": payload["git_revision_before"], "unique_texts": len(unique), "role_bindings": len(texts), "embedding_requests": embed_requests, "raw_requests": len(payload["requests"]), "model_state_after": payload["ps_after"]}))
