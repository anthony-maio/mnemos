"""Controlled development comparison for legacy content and access chronology."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import ipaddress
import json
from pathlib import Path
import socket
import time
from typing import Any

import httpx

import mnemos
import mnemos.engine as engine_module
from mnemos.config import MnemosConfig, MutableRAGConfig
from mnemos.engine import MnemosEngine, _build_scope_filter
from mnemos.types import MemoryChunk
from mnemos.utils.embeddings import SimpleEmbeddingProvider, cosine_similarity
from mnemos.utils.llm import MockLLMProvider
from mnemos.utils.storage import InMemoryStore, MemoryStore, SQLiteStore

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
MODES = ("ordered", "known-rewrite", "tied", "unknown-rewrite")
SCOPE = "chronology-contract"


def digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def stamp(seconds: int) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat()


def prepare(dataset: Path, destination: Path) -> dict[str, Any]:
    """Freeze the known fixture's chronology overlay without retrieving answers."""
    assert not destination.exists()
    documents = [json.loads(line) for line in dataset.read_text().splitlines() if line]
    assert len(documents) == 8
    assert [d["id"].split("-")[-1] for d in documents] == ["old", "new"] * 4
    provider = SimpleEmbeddingProvider(dim=384)
    originals = []
    for doc in documents:
        originals.append(
            {
                "id": doc["id"],
                "content": doc["content"],
                "embedding": provider.embed(doc["content"]),
                "metadata": {"scope": "project", "scope_id": SCOPE},
                "salience": 0.5,
                "cognitive_state": None,
                "access_count": 0,
                "version": 1,
            }
        )
    global_content = "A project guide is available for inspection."
    global_vector = provider.embed(global_content)
    state: dict[str, Any] = {"doc_freq": dict(provider._doc_freq), "doc_count": provider._doc_count}
    queries: list[dict[str, Any]] = []
    for doc in documents:
        for query in doc["queries"]:
            provider._doc_freq = Counter(state["doc_freq"])
            provider._doc_count = state["doc_count"]
            queries.append(
                {
                    "id": f"q{len(queries)}",
                    "text": query,
                    "original_expected_ids": [doc["id"]],
                    "embedding": provider.embed(query),
                }
            )
    assert len(queries) == 8 and len({q["text"] for q in queries}) == 8
    conditions = {}
    for mode in MODES:
        rows = []
        for index, original in enumerate(originals):
            row = copy.deepcopy(original)
            is_new = row["id"].endswith("-new")
            created = index
            revised = None
            updated = index
            if mode == "known-rewrite":
                created = -20 + index if is_new else index
                revised = 20 + index if is_new else None
                updated = revised if revised is not None else created
                row["version"] = 2 if is_new else 1
            elif mode == "tied":
                created = updated = 0
                revised = 0
            elif mode == "unknown-rewrite":
                created = -20 + index if is_new else index
                updated = 20 + index if is_new else created
                row["version"] = 2 if is_new else 1
            row.update(
                created_at=stamp(created),
                updated_at=stamp(updated),
                content_updated_at=stamp(revised) if revised is not None else None,
            )
            rows.append(row)
        for original in originals:
            row = copy.deepcopy(original)
            row.update(
                id="foreign-" + original["id"],
                created_at=stamp(500),
                updated_at=stamp(500),
                content_updated_at=stamp(500),
            )
            row["metadata"]["scope_id"] = "foreign-project"
            rows.append(row)
        workspace = copy.deepcopy(rows[0])
        workspace.update(
            id="workspace-control",
            created_at=stamp(500),
            updated_at=stamp(500),
            content_updated_at=stamp(500),
        )
        workspace["metadata"] = {"scope": "workspace", "scope_id": SCOPE}
        rows.append(workspace)
        rows.append(
            {
                "id": "global-control",
                "content": global_content,
                "embedding": global_vector,
                "metadata": {"scope": "global"},
                "salience": 0.5,
                "cognitive_state": None,
                "access_count": 0,
                "version": 1,
                "created_at": stamp(0),
                "updated_at": stamp(0),
                "content_updated_at": None,
            }
        )
        conditions[mode] = rows
    payload = {
        "schema": "mnemos-legacy-chronology-v1",
        "source_dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "source_documents": documents,
        "queries": queries,
        "embedding_dimension": 384,
        "embedding_state": state,
        "conditions": conditions,
        "top_k": 1,
        "candidate_limit": 20,
        "context_byte_limit": 2048,
        "current_scope": "project",
        "scope_id": SCOPE,
        "allowed_scopes": ["project", "global"],
        "query_clock": stamp(1000),
        "prior_access_clock": stamp(100),
        "determinate_conditions": ["ordered", "known-rewrite"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


class MatchedEmbedder(SimpleEmbeddingProvider):
    def __init__(self, fixture: dict[str, Any]) -> None:
        super().__init__(dim=384)
        self.state = fixture["embedding_state"]
        self.expected = {query["text"]: query["embedding"] for query in fixture["queries"]}
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self._doc_freq = Counter(self.state["doc_freq"])
        self._doc_count = self.state["doc_count"]
        result = super().embed(text)
        assert result == self.expected[text]
        self.calls += 1
        return result


def projection(store: MemoryStore) -> dict[str, Any]:
    common = []
    revisions = []
    for chunk in sorted(store.get_all(), key=lambda value: value.id):
        row = chunk.model_dump(mode="json")
        supported = "content_updated_at" in row
        revision = row.pop("content_updated_at", None)
        row["embedding"] = {
            "sha256": digest(row["embedding"]),
            "dimension": len(chunk.embedding or []),
        }
        common.append(row)
        revisions.append({"id": chunk.id, "supported": supported, "value": revision})
    return {"common": common, "common_sha256": digest(common), "revisions": revisions}


def block_external_calls() -> None:
    def deny(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("External provider calls are disabled for this comparison")

    original = socket.socket.connect

    def connect(sock: socket.socket, address: Any) -> Any:
        if not isinstance(address, tuple) or not ipaddress.ip_address(address[0]).is_loopback:
            return deny()
        return original(sock, address)

    setattr(socket.socket, "connect", connect)
    socket.create_connection = deny
    httpx.Client.send = deny  # type: ignore[method-assign]
    httpx.AsyncClient.send = deny  # type: ignore[method-assign]


async def measure(
    fixture: dict[str, Any], route: str, repetition: int, destination: Path
) -> dict[str, Any]:
    assert not destination.exists()
    destination.mkdir(parents=True)
    version = importlib.metadata.version("mnemos-memory")
    assert version == ("0.6.1" if route == "engine-061" else "0.6.2")
    import sys

    assert Path(sys.prefix).resolve() in Path(mnemos.__file__).resolve().parents
    fixed = datetime.fromisoformat(fixture["query_clock"])

    class FixedClock:
        @staticmethod
        def now(tz: Any = None) -> datetime:
            return fixed

    setattr(engine_module, "datetime", FixedClock)
    scope_filter = _build_scope_filter(
        current_scope="project", scope_id=SCOPE, allowed_scopes=("project", "global")
    )
    records: list[dict[str, Any]] = []
    access_order = [True, False] if repetition == 1 else [False, True]
    for backend in ["memory", "sqlite"]:
        for mode in MODES:
            for prior_access in access_order:
                path = destination / f"{backend}-{mode}-{int(prior_access)}.sqlite"
                store: MemoryStore = (
                    InMemoryStore() if backend == "memory" else SQLiteStore(str(path))
                )
                try:
                    for query in fixture["queries"]:
                        for row in fixture["conditions"][mode]:
                            store.store(MemoryChunk.model_validate(copy.deepcopy(row)))
                        if prior_access:
                            for row in fixture["conditions"][mode][:8]:
                                if row["id"].endswith("-old"):
                                    assert store.touch(
                                        row["id"],
                                        updated_at=datetime.fromisoformat(
                                            fixture["prior_access_clock"]
                                        ),
                                    )
                        before = projection(store)
                        embedder = MatchedEmbedder(fixture)
                        engine = MnemosEngine(
                            config=MnemosConfig(mutable_rag=MutableRAGConfig(enabled=False)),
                            llm=MockLLMProvider(),
                            embedder=embedder,
                            store=store,
                        )
                        trace: list[dict[str, Any]] = []
                        original_lookup = store.retrieve

                        def observed(
                            embedding: list[float], top_k: int = 5, filter_fn: Any = None
                        ) -> list[MemoryChunk]:
                            chunks = original_lookup(embedding, top_k=top_k, filter_fn=filter_fn)
                            trace.append(
                                {
                                    "top_k": top_k,
                                    "query_vector_sha256": digest(embedding),
                                    "candidates": [
                                        {
                                            "id": chunk.id,
                                            "cosine": cosine_similarity(
                                                embedding, chunk.embedding or []
                                            ),
                                        }
                                        for chunk in chunks
                                    ],
                                }
                            )
                            return chunks

                        setattr(store, "retrieve", observed)
                        result: dict[str, Any] = {
                            "route": route,
                            "repetition": repetition,
                            "backend": backend,
                            "mode": mode,
                            "prior_access": prior_access,
                            "query_id": query["id"],
                            "before": before,
                            "trace": trace,
                        }
                        start = time.perf_counter()
                        try:
                            if route == "vector-062":
                                vector = embedder.embed(query["text"])
                                returned = store.retrieve(vector, top_k=20, filter_fn=scope_filter)[
                                    :1
                                ]
                            else:
                                returned = await engine.retrieve(
                                    query["text"],
                                    top_k=1,
                                    reconsolidate=False,
                                    current_scope="project",
                                    scope_id=SCOPE,
                                    allowed_scopes=("project", "global"),
                                )
                            context = "\n".join(
                                f"fact_id={chunk.id}\ncontent={chunk.content}" for chunk in returned
                            )
                            result.update(
                                status="returned",
                                returned_ids=[chunk.id for chunk in returned],
                                context=context,
                                context_bytes=len(context.encode("utf-8")),
                            )
                        except Exception as error:
                            result.update(
                                status="error", error_type=type(error).__name__, error=str(error)
                            )
                        finally:
                            result["elapsed_ms"] = (time.perf_counter() - start) * 1000
                            setattr(store, "retrieve", original_lookup)
                        result.update(after=projection(store), embedding_calls=embedder.calls)
                        records.append(result)
                finally:
                    if isinstance(store, SQLiteStore):
                        store.close()
    result = {
        "schema": "mnemos-legacy-chronology-result-v1",
        "route": route,
        "repetition": repetition,
        "installed_version": version,
        "fixture_sha256": digest(fixture),
        "records": records,
        "external_providers_disabled": True,
        "fresh_acceptance": False,
    }
    (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--fixture", type=Path, required=True)
    run.add_argument("--route", choices=["engine-061", "engine-062", "vector-062"], required=True)
    run.add_argument("--repetition", type=int, choices=[0, 1, 2], required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    block_external_calls()
    if args.command == "prepare":
        result = prepare(args.dataset, args.output)
        print(
            json.dumps(
                {
                    "prepared": True,
                    "queries": len(result["queries"]),
                    "records_per_condition": len(result["conditions"]["ordered"]),
                }
            )
        )
    else:
        fixture = json.loads(args.fixture.read_bytes())
        result = asyncio.run(measure(fixture, args.route, args.repetition, args.output))
        print(
            json.dumps(
                {
                    "records": len(result["records"]),
                    "route": args.route,
                    "errors": sum(row["status"] == "error" for row in result["records"]),
                }
            )
        )


if __name__ == "__main__":
    main()
