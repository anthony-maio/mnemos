"""Compare loaded context across Ollama endpoints without changing existing aliases."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any
import urllib.request

OLD = "mnemos-native-qwen35-9b:greedy-v1"
OLD_DIGEST = "243c7dc812e847df8f175763c8e91b6daa88483f141957b76f9e9722e8d526a5"
NEW = "mnemos-native-qwen35-9b:greedy-ctx64k-v2"
CONTEXT = 65536


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runner_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "requests": [],
        "observations": [],
    }

    def save() -> None:
        (output / "report.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    def request(label: str, route: str, payload: Any = None) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        if body is not None:
            (output / (label + ".request")).write_bytes(body)
        req = urllib.request.Request(
            "http://127.0.0.1:11434" + route,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        row: dict[str, Any] = {"label": label, "route": route, "method": req.method}
        record["requests"].append(row)
        save()
        started = time.monotonic()
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=180 if "chat" in route else 30) as response:
                raw = response.read()
                row["http_status"] = response.status
            (output / (label + ".response")).write_bytes(raw)
            row["response_sha256"] = hashlib.sha256(raw).hexdigest()
            return json.loads(raw)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            row["seconds"] = time.monotonic() - started
            save()

    def observe(label: str, model: str, route: str) -> None:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with READY."}],
            "stream": False,
            "think": False,
            "options": {"num_ctx": CONTEXT},
        }
        if route == "/api/chat":
            payload["options"]["num_predict"] = 16
        else:
            payload.update(max_tokens=16, reasoning_effort="none")
        response = request(label, route, payload)
        loaded = request(label + "-ps", "/api/ps")
        rows = [r for r in loaded["models"] if r.get("name") == model]
        if len(rows) != 1 or not rows[0].get("size_vram", 0) > 0:
            raise ValueError("Expected one loaded model with positive GPU residency")
        row = rows[0]
        record["observations"].append(
            {
                "label": label,
                "model": model,
                "route": route,
                "requested_context": CONTEXT,
                "loaded_context": row.get("context_length"),
                "digest": row["digest"],
                "size_vram": row["size_vram"],
                "response_complete": (
                    response.get("done") if route == "/api/chat" else bool(response.get("choices"))
                ),
            }
        )
        save()
        print(json.dumps(record["observations"][-1]), flush=True)

    try:
        request("version", "/api/version")
        tags = request("tags-before", "/api/tags")
        models = {row["name"]: row["digest"] for row in tags["models"]}
        if models.get(OLD) != OLD_DIGEST or NEW in models:
            raise ValueError("Old alias changed or prospective alias already exists")
        old_show = request("old-show", "/api/show", {"model": OLD})
        request("ps-before", "/api/ps")
        observe("old-native-api", OLD, "/api/chat")
        observe("old-openai-api", OLD, "/v1/chat/completions")
        request(
            "create-context-alias",
            "/api/create",
            {
                "model": NEW,
                "from": OLD,
                "parameters": {"num_ctx": CONTEXT},
                "stream": False,
            },
        )
        new_show = request("new-show", "/api/show", {"model": NEW})
        params = dict(line.split(None, 1) for line in new_show["parameters"].splitlines())
        if any(
            float(params.get(key, "nan")) != value
            for key, value in {
                "num_ctx": CONTEXT,
                "temperature": 0,
                "seed": 42,
                "presence_penalty": 0,
                "top_k": 20,
                "top_p": 0.95,
            }.items()
        ):
            raise ValueError("New alias does not preserve the intended profile")
        for key in ("template", "model_info", "capabilities"):
            if old_show.get(key) != new_show.get(key):
                raise ValueError("New alias changed " + key)
        old_from = next(
            line for line in old_show["modelfile"].splitlines() if line.startswith("FROM ")
        )
        new_from = next(
            line for line in new_show["modelfile"].splitlines() if line.startswith("FROM ")
        )
        if old_from != new_from:
            raise ValueError("New alias changed model weight blob")
        observe("new-openai-api", NEW, "/v1/chat/completions")
        after = request("tags-after", "/api/tags")
        final = {row["name"]: row["digest"] for row in after["models"]}
        if any(final.get(name) != digest for name, digest in models.items()):
            raise ValueError("An existing model changed during the probe")
        if any(final.get(row["model"]) != row["digest"] for row in record["observations"]):
            raise ValueError("Loaded model digest differs from the final model inventory")
        record.update(
            status="completed",
            new_digest=final[NEW],
            existing_models_unchanged=True,
            same_weight_blob=old_from,
            parameters=params,
            loaded_contexts=[row["loaded_context"] for row in record["observations"]],
            context_restored=record["observations"][-1]["loaded_context"] == CONTEXT,
        )
        return 0
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    raise SystemExit(main())
