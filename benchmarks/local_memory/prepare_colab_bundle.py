"""Freeze a held-out run for a cloud GPU: source, corpus, models, and settings.

Run this where the repository and the Ollama models live. It writes a bundle
directory to upload to Google Drive, where the Colab notebook in
benchmarks/local_memory/colab picks it up. The source comes from ``git
archive`` of a clean HEAD, so the bundle is exactly one commit. The model
digests come from the local Ollama server, so Colab can refuse to compare
different weights. The context length is pinned because Ollama sizes its
default from VRAM, and a bigger GPU would otherwise run a different context.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

from .contracts import load_dataset
from .harness import DEFAULT_PROFILES
from .heldout_run import _live_model_metadata

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

# The runnable surface the notebook needs. Validation archives stay behind.
ARCHIVED_PATHS = ("mnemos", "benchmarks", "pyproject.toml")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def build_bundle(
    *,
    repo: Path,
    corpus: Path,
    output: Path,
    endpoint: str,
    model: str,
    embedding_model: str,
    context_length: int,
    repetitions: int,
    max_tokens: int,
    timeout_seconds: float,
    run_namespace: str | None,
) -> dict[str, object]:
    if _git(repo, "status", "--porcelain", "--untracked-files=no", "--", *ARCHIVED_PATHS):
        raise RuntimeError("commit the source first; the bundle archives HEAD only")
    commit = _git(repo, "rev-parse", "HEAD")
    dataset = load_dataset(corpus)
    if len(dataset.heldout_sessions) < 200:
        raise ValueError("held-out corpus has fewer than 200 sessions")
    models = _live_model_metadata(endpoint, (model, embedding_model))
    with urlopen(endpoint.rstrip("/") + "/api/version", timeout=30.0) as response:  # noqa: S310
        ollama_version = str(json.loads(response.read().decode("utf-8"))["version"])

    output.mkdir(parents=True, exist_ok=False)
    archive = subprocess.run(
        ["git", "archive", "--format=zip", "HEAD", *ARCHIVED_PATHS],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout
    (output / "source.zip").write_bytes(archive)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle_zip:
        source_files = {
            info.filename: hashlib.sha256(bundle_zip.read(info)).hexdigest()
            for info in bundle_zip.infolist()
            if not info.is_dir()
        }
    shutil.copyfile(corpus, output / "corpus.jsonl")
    pyproject = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    preparation: dict[str, object] = {
        "source_commit": commit,
        "source_files": source_files,
        "corpus_sha256": hashlib.sha256((output / "corpus.jsonl").read_bytes()).hexdigest(),
        "dataset_version": dataset.version,
        "heldout_sessions": len(dataset.heldout_sessions),
        "repetitions": repetitions,
        "profiles": [profile.name for profile in DEFAULT_PROFILES],
        "model": model,
        "embedding_model": embedding_model,
        "models": models,
        "ollama_version": ollama_version,
        "ollama_context_length": context_length,
        "options": {
            "think": False,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "timeout_seconds": timeout_seconds,
            "extraction_mode": "source_bound",
        },
        "run_namespace": run_namespace or f"colab-{commit[:12]}",
        "python_dependencies": list(pyproject["project"]["dependencies"]),
    }
    (output / "preparation.json").write_text(
        json.dumps(preparation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return preparation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    # What Ollama 0.34.2 chose for qwen3.5:9b on the 16 GB card the earlier
    # held-out run used.
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--run-namespace")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    preparation = build_bundle(
        repo=repo,
        corpus=args.corpus,
        output=args.output,
        endpoint=args.endpoint,
        model=args.model,
        embedding_model=args.embedding_model,
        context_length=args.context_length,
        repetitions=args.repetitions,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        run_namespace=args.run_namespace,
    )
    print(
        json.dumps(
            {
                "bundle": str(args.output),
                "source_commit": preparation["source_commit"],
                "dataset_version": preparation["dataset_version"],
                "heldout_sessions": preparation["heldout_sessions"],
                "next": "upload this folder to MyDrive/mnemos-eval/ and open the notebook",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
