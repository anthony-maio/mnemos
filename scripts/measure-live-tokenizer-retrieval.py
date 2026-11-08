"""Execute the fixed live-tokenizer schedule against frozen canonical source."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


async def run(root, identity_path):
    freeze = read(root / "freeze.json")
    for path, digest in freeze["bindings"].items():
        assert sha(Path(path)) == digest, path
    actual_versions = {
        d.metadata["Name"].lower().replace("_", "-"): d.version
        for d in importlib.metadata.distributions()
    }
    assert actual_versions == freeze["versions"]
    canonical = Path(freeze["canonical_path"])
    sys.path.insert(0, str(canonical))
    import mnemos

    assert Path(mnemos.__file__).resolve().parent == canonical / "mnemos"
    sys.path.insert(0, freeze["benchmark_module_path"])
    from live_tokenizer import LiveTokenCounter
    from live_tokenizer_retrieval import observe_retrieval
    from mnemos.context_budget import HostTokenizer
    from mnemos.utils.embeddings import FastembedProvider

    identity = read(identity_path)
    assert identity["model_blob_sha256"] == freeze["model_blob_sha256"]
    assert identity["native_binary_sha256"] == freeze["native_binary_sha256"]
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        FASTEMBED_CACHE_PATH=freeze["embedding_cache"],
    )
    output = root / "measurement"
    output.mkdir(exist_ok=False)
    (output / "stores").mkdir()
    counter = LiveTokenCounter(identity["tokenizer_url"])
    try:
        started = time.perf_counter()
        embedder = FastembedProvider(cache_dir=freeze["embedding_cache"])
        embedder.embed_query("Tokenizer integration initialization control")
        (output / "initialization.json").write_text(
            json.dumps(
                {
                    "embedding_initialization_ms": (time.perf_counter() - started) * 1000,
                    "identity_sha256": sha(identity_path),
                    "pid": os.getpid(),
                }
            )
        )
        cases = read(Path(freeze["scenarios"]))
        inputs = sorted(read(Path(freeze["inputs"])), key=lambda row: row["key"])
        assert len(inputs) == 68 and len(cases) == 32
        native = HostTokenizer(
            "qwen3:14b-native-literal", identity["model_blob_sha256"], counter.count
        )
        injected = []

        def broken(text):
            injected.append(text)
            raise RuntimeError("Prospective injected counter failure")

        fallback = HostTokenizer(native.name, native.revision, broken)
        with (
            (output / "starts.jsonl").open("x") as starts,
            (output / "observations.jsonl").open("x") as records,
        ):
            index = 0
            for query_index, query in enumerate(inputs):
                arms = ["native", "bytes", "failed"]
                arms = arms[query_index % 3 :] + arms[: query_index % 3]
                for budget in [64, 2048]:
                    for arm in arms:
                        entry = {
                            "index": index,
                            "key": query["key"],
                            "case_key": query["case_key"],
                            "budget": budget,
                            "arm": arm,
                        }
                        starts.write(json.dumps(entry) + "\n")
                        starts.flush()
                        counter.records.clear()
                        injected.clear()
                        try:
                            result = await observe_retrieval(
                                cases[query["case_key"]],
                                output / "stores" / f"{index:03d}.sqlite",
                                query=query["query"],
                                budget=budget,
                                embedder=embedder,
                                tokenizer={"native": native, "bytes": None, "failed": fallback}[
                                    arm
                                ],
                            )
                            calls = list(counter.records)
                            context = result["result"]["context"]
                            actual_count = counter.count(context)
                            native_reply = json.loads(counter.records[-1]["response_text"])
                            response = counter.client.post(
                                "/detokenize", json={"tokens": native_reply["tokens"]}
                            )
                            response.raise_for_status()
                            assert response.json().get("content") == context
                            metadata = result["result"]
                            if arm == "native":
                                assert metadata["token_count_method"] == "host_tokenizer"
                                assert metadata["context_token_count"] == actual_count <= budget
                            else:
                                assert metadata["token_count_method"] == "utf8_bytes_upper_bound"
                                assert metadata["context_token_count"] is None
                                assert (
                                    metadata["context_token_upper_bound"]
                                    == len(context.encode("utf-8"))
                                    <= budget
                                )
                            entry.update(
                                status="assessed",
                                **result,
                                callback_records=calls,
                                injected_inputs=list(injected),
                                final_context_native_tokens=actual_count,
                                audit_tokenization=counter.records[-1],
                                audit_detokenization=response.text,
                            )
                        except Exception as exc:
                            entry.update(
                                status="unavailable",
                                error_type=type(exc).__name__,
                                error=str(exc),
                                callback_records=list(counter.records),
                                injected_inputs=list(injected),
                            )
                        records.write(json.dumps(entry, ensure_ascii=True) + "\n")
                        records.flush()
                        index += 1
            assert index == 408
        for path, digest in freeze["bindings"].items():
            assert sha(Path(path)) == digest, path
        (output / "complete.json").write_text(
            json.dumps({"observations": index, "bindings_unchanged": True})
        )
        print("Completed 408 observations; source and input bindings unchanged.", flush=True)
    finally:
        counter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.root.resolve(), args.runtime_receipt.resolve()))
