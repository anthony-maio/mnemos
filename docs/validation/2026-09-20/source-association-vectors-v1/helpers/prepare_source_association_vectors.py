"""Prepare a pinned source-association embedding artifact from local Ollama."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ENDPOINT = "http://127.0.0.1:11434"
MODEL = "nomic-embed-text:latest"
DIGEST = "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f"
DIMENSION = 768
BATCH_SIZE = 16
TIMEOUT_SECONDS = 60
SCHEMA_VERSION = "source-association-vectors-v1"
Transport = Callable[[str, str, bytes | None, int], tuple[int, dict[str, str], bytes]]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def manifest_texts(manifest: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    cases = manifest.get("graph_cases", manifest.get("cases"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest cases must be a nonempty list")
    unique: dict[str, None] = {}
    bindings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each case must be an object")
        case_id = _text(case.get("id"), "case id")
        if case_id in seen:
            raise ValueError(f"duplicate case id: {case_id}")
        seen.add(case_id)
        query = _text(case.get("query"), f"{case_id}.query")
        unique[query] = None
        bindings.append({"case_id": case_id, "role": "query", "text": query})
        for field, role in (
            ("facts", "fact"),
            ("events", "event"),
            ("scope_controls", "scope_control"),
        ):
            rows = case.get(field, []) if field == "scope_controls" else case.get(field)
            if not isinstance(rows, list):
                raise ValueError(f"{case_id}.{field} must be a list")
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f"{case_id}.{field} rows must be objects")
                row_id = _text(row.get("id"), f"{case_id}.{field}.id")
                content = _text(row.get("content"), f"{case_id}.{field}.content")
                unique[content] = None
                bindings.append({"case_id": case_id, "role": role, "id": row_id, "text": content})
        actions = case.get("lifecycle_actions", [])
        if not isinstance(actions, list):
            raise ValueError(f"{case_id}.lifecycle_actions must be a list")
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                raise ValueError(f"{case_id}.lifecycle_actions[{index}] must be an object")
            correction_value = action.get("correction")
            correction: dict[str, Any] = (
                correction_value if isinstance(correction_value, dict) else action
            )
            if "new_content" in correction:
                content = _text(
                    correction["new_content"], f"{case_id}.lifecycle_actions.new_content"
                )
                unique[content] = None
                bindings.append(
                    {
                        "case_id": case_id,
                        "role": "correction_new_content",
                        "action_index": index,
                        "text": content,
                    }
                )
    return list(unique), bindings


def load_manifest(path: Path) -> tuple[dict[str, Any], str, list[str], list[dict[str, Any]]]:
    raw = path.read_bytes()
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    texts, bindings = manifest_texts(manifest)
    return manifest, sha256_bytes(raw), texts, bindings


def _transport(
    method: str, url: str, body: bytes | None, timeout: int
) -> tuple[int, dict[str, str], bytes]:
    request = Request(url, data=body, method=method, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(response.status), dict(response.headers.items()), response.read()
    except HTTPError as exc:
        return int(exc.code), dict(exc.headers.items()), exc.read()
    except URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc.reason}") from exc


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("git repository validation failed") from exc


def _committed_inputs(repo: Path, paths: list[Path]) -> tuple[Path, str, dict[str, str]]:
    top = Path(_git(repo, "rev-parse", "--show-toplevel")).resolve()
    head = _git(top, "rev-parse", "HEAD")
    if not head:
        raise ValueError("git HEAD is required")
    hashes: dict[str, str] = {}
    for path in paths:
        path = path.resolve()
        if not path.is_relative_to(top):
            raise ValueError("manifest and helper must share the repository")
        relative = path.relative_to(top).as_posix()
        try:
            tracked = _git(top, "ls-files", "--error-unmatch", "--", relative)
            committed = subprocess.check_output(
                ["git", "-C", str(top), "show", f"{head}:{relative}"]
            )
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            raise ValueError(f"input is not committed at HEAD: {relative}") from exc
        if tracked != relative:
            raise ValueError(f"input tracking mismatch: {relative}")
        current = path.read_bytes()
        if current != committed and current.replace(b"\r\n", b"\n") != committed.replace(
            b"\r\n", b"\n"
        ):
            raise ValueError(f"working input differs from committed HEAD: {relative}")
        hashes[relative] = sha256_bytes(current)
    return top, head, hashes


def _digest(tags: Any) -> str:
    models = tags.get("models") if isinstance(tags, dict) else None
    matches = [row for row in models or [] if isinstance(row, dict) and row.get("name") == MODEL]
    if len(matches) != 1 or matches[0].get("digest") != DIGEST:
        raise ValueError("pinned embedding model digest mismatch")
    return DIGEST


def _validate_show(show: Any) -> None:
    if not isinstance(show, dict) or not isinstance(show.get("details"), dict):
        raise ValueError("Ollama show response lacks model details")
    if show["details"].get("family") != "nomic-bert":
        raise ValueError("Ollama show response is not the pinned embedding family")
    if not isinstance(show.get("model_info"), dict) or not isinstance(show.get("parameters"), str):
        raise ValueError("Ollama show response lacks model configuration")


def _show_signature(show: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in show.items() if key != "modified_at"}


def _vectors(response: Any, count: int) -> list[list[float]]:
    rows = response.get("embeddings") if isinstance(response, dict) else None
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("embedding response count does not match request batch")
    result: list[list[float]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != DIMENSION:
            raise ValueError("embedding dimension does not match pinned 768")
        if any(
            not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(float(x))
            for x in row
        ):
            raise ValueError("embedding contains a nonfinite value")
        if not any(float(x) != 0.0 for x in row):
            raise ValueError("embedding vector is all zero")
        result.append([float(x) for x in row])
    return result


def prepare_source_association_vectors(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    endpoint: str = ENDPOINT,
    transport: Transport | None = None,
    repo_root: str | Path | None = None,
    helper_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    repo_hint = Path(repo_root or manifest_path.parent).resolve()
    helper = Path(helper_path or __file__).resolve()
    repo, git_before, committed_hashes = _committed_inputs(repo_hint, [manifest_path, helper])
    if output == repo or repo in output.parents:
        raise ValueError("output must be outside the repository")
    if endpoint.rstrip("/") != ENDPOINT:
        raise ValueError("preparation is restricted to the pinned loopback endpoint")
    output.mkdir(parents=True)
    raw_dir = output / "raw"
    raw_dir.mkdir()
    try:
        _manifest, manifest_sha, texts, bindings = load_manifest(manifest_path)
    except Exception as exc:
        (output / "artifact.json").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "manifest": {"path": str(manifest_path)},
                    "input_hashes": {"helper": sha256_bytes(helper.read_bytes())},
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "input_hashes": committed_hashes,
        "git_revision_before": git_before,
        "model": MODEL,
        "expected_digest": DIGEST,
        "endpoint": endpoint.rstrip("/"),
        "dimension": DIMENSION,
        "batch_size": BATCH_SIZE,
        "request_timeout_seconds": TIMEOUT_SECONDS,
        "embedding_text_contract": {
            "query_prefix": "",
            "document_prefix": "",
            "symmetric_prefix": "",
            "symmetry": "symmetric",
        },
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "requests": [],
        "vectors": {},
        "bindings": [],
    }
    report = output / "artifact.json"
    report.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    request = transport or _transport
    number = 0

    def call(method: str, path: str, payload: Any | None = None) -> Any:
        nonlocal number
        number += 1
        body = None if payload is None else _json_bytes(payload)
        stem = f"request-{number:04d}"
        if body is not None:
            (raw_dir / f"{stem}.request.bin").write_bytes(body)
        entry: dict[str, Any] = {
            "index": number,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "url": endpoint.rstrip("/") + path,
            "request_headers": {"content-type": "application/json"},
            "request_bytes": len(body or b""),
            "request_sha256": sha256_bytes(body) if body else None,
        }
        artifact["requests"].append(entry)
        report.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            status, headers, response = request(
                method, endpoint.rstrip("/") + path, body, TIMEOUT_SECONDS
            )
        except Exception as exc:
            entry.update(
                status="transport_error",
                error_type=type(exc).__name__,
                error=str(exc),
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            report.write_text(
                json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            raise
        (raw_dir / f"{stem}.response.bin").write_bytes(response)
        entry.update(
            response_status=status,
            response_bytes=len(response),
            response_sha256=sha256_bytes(response),
            finished_at=datetime.now(timezone.utc).isoformat(),
            response_headers={
                str(k).lower(): str(v)
                for k, v in headers.items()
                if str(k).lower() in {"content-type", "content-length"}
            },
        )
        report.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if status < 200 or status >= 300:
            raise RuntimeError(f"Ollama request returned HTTP {status}")
        try:
            return json.loads(response)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Ollama response was not JSON") from exc

    try:
        artifact["ollama_version"] = call("GET", "/api/version")
        artifact["tags_before"] = call("GET", "/api/tags")
        _digest(artifact["tags_before"])
        artifact["show_before"] = call("POST", "/api/show", {"model": MODEL})
        _validate_show(artifact["show_before"])
        artifact["ps_before"] = call("GET", "/api/ps")
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            response = call(
                "POST", "/api/embed", {"model": MODEL, "input": batch, "truncate": False}
            )
            for text, vector in zip(batch, _vectors(response, len(batch))):
                artifact["vectors"][sha256_bytes(text.encode())] = {"text": text, "vector": vector}
        artifact["tags_after"] = call("GET", "/api/tags")
        _digest(artifact["tags_after"])
        artifact["show_after"] = call("POST", "/api/show", {"model": MODEL})
        _validate_show(artifact["show_after"])
        if _show_signature(artifact["show_before"]) != _show_signature(artifact["show_after"]):
            raise RuntimeError("embedding model show metadata changed during preparation")
        artifact["ps_after"] = call("GET", "/api/ps")
        artifact["bindings"] = [
            {**row, "text_sha256": sha256_bytes(row["text"].encode())} for row in bindings
        ]
        if len(artifact["vectors"]) != len(texts):
            raise ValueError("vector map count does not match unique text count")
        _, git_after, hashes_after = _committed_inputs(repo, [manifest_path, helper])
        if hashes_after != committed_hashes:
            raise RuntimeError("committed input changed during preparation")
        artifact["git_revision_after"] = git_after
        if git_after != git_before:
            raise RuntimeError("git revision changed during preparation")
        artifact["status"] = "complete"
        artifact["finished_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        artifact.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        report.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    report.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=ENDPOINT)
    args = parser.parse_args()
    prepare_source_association_vectors(args.manifest, args.output, endpoint=args.endpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
