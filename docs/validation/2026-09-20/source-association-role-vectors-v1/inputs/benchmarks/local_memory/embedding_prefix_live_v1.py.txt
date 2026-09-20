"""Record one frozen local prefix diagnostic without retries or live memory writes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
from typing import Any

from mnemos.utils.embeddings import EmbeddingProvider
from .embedding_prefix_development_v1 import MANIFEST_PATH, run_paired_diagnostic
from .prepare_source_association_vectors import (
    BATCH_SIZE,
    DIGEST,
    DIMENSION,
    ENDPOINT,
    MODEL,
    TIMEOUT_SECONDS,
    Transport,
    _committed_inputs,
    _digest,
    _json_bytes,
    _show_signature,
    _transport,
    _validate_show,
    _vectors,
    sha256_bytes,
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RecordedAPI:
    def __init__(self, output: Path, *, transport: Transport = _transport) -> None:
        self.output = output
        self.raw = output / "raw"
        self.raw.mkdir(parents=True, exist_ok=False)
        self.transport = transport
        self.number = 0

    def call(self, arm: str, route: str, payload: Any = None) -> Any:
        self.number += 1
        stem = f"raw/request-{self.number:04d}"
        body = None if payload is None else _json_bytes(payload)
        row: dict[str, Any] = {
            "index": self.number,
            "arm": arm,
            "started_at": utcnow(),
            "method": "GET" if body is None else "POST",
            "url": ENDPOINT + route,
            "request_bytes": len(body or b""),
        }
        if body is not None:
            row.update(request_file=stem + ".request.bin", request_sha256=sha256_bytes(body))
            (self.output / row["request_file"]).write_bytes(body)
        try:
            status, headers, data = self.transport(row["method"], row["url"], body, TIMEOUT_SECONDS)
            row.update(
                response_status=status,
                response_file=stem + ".response.bin",
                response_sha256=sha256_bytes(data),
                response_bytes=len(data),
                response_headers={
                    k.lower(): v
                    for k, v in headers.items()
                    if k.lower() in {"content-type", "content-length"}
                },
            )
            (self.output / row["response_file"]).write_bytes(data)
            if not 200 <= status < 300:
                raise RuntimeError(f"Ollama request returned HTTP {status}")
            result = json.loads(data)
            row["status"] = "complete"
            return result
        except Exception as exc:
            row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            row["finished_at"] = utcnow()
            with (self.output / "requests.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")


class RecordedProvider(EmbeddingProvider):
    def __init__(self, api: RecordedAPI, arm: str) -> None:
        self.api = api
        self.arm = arm
        self.loaded_profiles: list[dict[str, Any]] = []

    @property
    def dim(self) -> int:
        return DIMENSION

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            response = self.api.call(
                self.arm, "/api/embed", {"model": MODEL, "input": batch, "truncate": False}
            )
            vectors = _vectors(response, len(batch))
            loaded = self.api.call(self.arm, "/api/ps")
            matches = [m for m in loaded.get("models", []) if m.get("name") == MODEL]
            if len(matches) != 1 or matches[0].get("digest") != DIGEST:
                raise ValueError("loaded embedding model digest mismatch")
            profile = matches[0]
            gpu = profile.get("size_vram")
            if (
                type(gpu) not in {int, float}
                or not math.isfinite(gpu)
                or gpu <= 0
                or type(profile.get("context_length")) is not int
                or profile["context_length"] <= 0
            ):
                raise ValueError("embedding model GPU/context observation is invalid")
            self.loaded_profiles.append(profile)
            result.extend(vectors)
        return result


def run(output: Path) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    protocol = repo / "docs/plans/2026-09-20-embedding-prefix-development-v1.md"
    paths = [
        MANIFEST_PATH,
        protocol,
        Path(__file__),
        Path(__file__).with_name("embedding_prefix_development_v1.py"),
        Path(__file__).with_name("prepare_source_association_vectors.py"),
        repo / "mnemos/utils/embeddings.py",
        repo / "mnemos/scopes.py",
        repo / "mnemos/types.py",
    ]
    repo, commit, hashes = _committed_inputs(repo, paths)
    output = output.resolve()
    if output.is_relative_to(repo):
        raise ValueError("diagnostic output must be outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    api = RecordedAPI(output)
    report: dict[str, Any] = {
        "status": "running",
        "started_at": utcnow(),
        "source_commit": commit,
        "input_hashes": hashes,
        "endpoint": ENDPOINT,
        "model": MODEL,
        "digest": DIGEST,
        "dimension": DIMENSION,
        "batch_size": BATCH_SIZE,
        "truncate": False,
        "retries": 0,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pydantic")},
        },
    }
    report_path = output / "artifact.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    providers: dict[str, RecordedProvider] = {}
    try:
        report["ollama_version"] = api.call("metadata", "/api/version")
        report["tags_before"] = api.call("metadata", "/api/tags")
        _digest(report["tags_before"])
        report["show_before"] = api.call("metadata", "/api/show", {"model": MODEL})
        _validate_show(report["show_before"])
        report["ps_before"] = api.call("metadata", "/api/ps")

        def factory(label: str) -> RecordedProvider:
            if label in providers:
                raise ValueError("duplicate arm")
            providers[label] = RecordedProvider(api, label)
            return providers[label]

        run_paired_diagnostic(
            factory,
            output / "paired.json",
            model_metadata={
                "model": MODEL,
                "digest": DIGEST,
                "show": report["show_before"],
                "ollama_version": report["ollama_version"],
            },
            code_sha=sha256_bytes(_json_bytes(hashes)),
            options={
                "endpoint": ENDPOINT,
                "batch_size": BATCH_SIZE,
                "truncate": False,
                "request_timeout_seconds": TIMEOUT_SECONDS,
                "retries": 0,
                "protocol_sha256": sha256_bytes(protocol.read_bytes()),
            },
        )
        report["tags_after"] = api.call("metadata", "/api/tags")
        _digest(report["tags_after"])
        report["show_after"] = api.call("metadata", "/api/show", {"model": MODEL})
        _validate_show(report["show_after"])
        if _show_signature(report["show_before"]) != _show_signature(report["show_after"]):
            raise ValueError("model configuration changed during diagnostic")
        report["ps_after"] = api.call("metadata", "/api/ps")
        _, after_commit, after_hashes = _committed_inputs(repo, paths)
        if (after_commit, after_hashes) != (commit, hashes):
            raise ValueError("committed inputs changed during diagnostic")
        report["paired_sha256"] = sha256_bytes((output / "paired.json").read_bytes())
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["finished_at"] = utcnow()
        report["request_count"] = api.number
        report["loaded_profiles"] = {
            label: provider.loaded_profiles for label, provider in providers.items()
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"status": result["status"], "requests": result["request_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
