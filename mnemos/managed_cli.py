"""CLI facade for the single local-memory service."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from argparse import Namespace
from typing import Any

from .client import MnemosClient
from .scopes import ScopeContext, ScopeRegistry, default_registry_path


def resolve_context() -> ScopeContext:
    profile = os.environ.get("MNEMOS_USER_PROFILE_ID")
    if not profile:
        raise ValueError("Set MNEMOS_USER_PROFILE_ID and enable the project before using memory")
    registry = ScopeRegistry.load(default_registry_path())
    return registry.resolve(
        os.environ.get("MNEMOS_PROJECT_ROOT", str(Path.cwd())),
        user_profile_id=profile,
        session_id=os.environ.get("MNEMOS_SESSION_ID") or str(uuid4()),
        host="manual",
        workspace_id=os.environ.get("MNEMOS_WORKSPACE_ID"),
    )


async def run_command(args: Namespace) -> dict[str, Any]:
    response: dict[str, Any]
    command = args.command
    client = MnemosClient(auto_start=False if command == "service-stop" else None)
    if command == "service-stop":
        from .service import read_runtime_descriptor

        descriptor = read_runtime_descriptor(client.descriptor_path)
        if descriptor is None:
            response = {"schema_version": 1, "status": "not_running", "committed": False}
        else:
            response = await client.request("shutdown", {})
            if response.get("status") == "stopping":
                for _ in range(100):
                    await asyncio.sleep(0.1)
                    current = read_runtime_descriptor(client.descriptor_path)
                    if current is None or current.get("token") != descriptor.get("token"):
                        response["status"] = "stopped"
                        break
                else:
                    response.update(status="degraded", error="shutdown_not_confirmed")
    elif command == "enable-project":
        response = await client.request(
            "configure_project",
            {
                "path": str(Path(args.path).resolve()),
                "user_profile_id": args.profile,
                "enabled": True,
                "workspace_id": args.workspace_id,
            },
        )
    elif command in {"backup", "restore"}:
        payload = {"destination": str(Path(args.destination).resolve())}
        if command == "restore":
            payload["archive"] = str(Path(args.archive).resolve())
        response = await client.request(command, payload)
    elif command in {"stats", "consolidate"}:
        response = await client.request("health" if command == "stats" else "maintenance", {})
    else:
        context = resolve_context()
        data: dict[str, Any] = {"context": context.model_dump()}
        supplied_id = getattr(args, "scope_id", None)
        current_scope = getattr(args, "scope", None) or getattr(args, "current_scope", "project")
        expected_id = context.workspace_id if current_scope == "workspace" else context.project_id
        if supplied_id not in {None, "", "default", expected_id}:
            raise ValueError("scope-id conflicts with the registered project or workspace")
        if command == "jobs":
            data.update(status=args.status, limit=args.limit)
            response = await client.request("jobs", data)
        elif command == "legacy-review":
            data.update(limit=args.limit, after_id=args.after_id)
            response = await client.request("legacy_review", data)
        elif command in {"job-status", "retry-job"}:
            data["job_id"] = args.job_id
            response = await client.request("job" if command == "job-status" else "retry_job", data)
        elif command == "store":
            data.update(content=args.content, scope=args.scope, role=args.role)
            response = await client.request("remember", data)
        elif command == "retrieve":
            allowed = [part.strip() for part in args.allowed_scopes.split(",") if part.strip()]
            data.update(
                query=args.query,
                top_k=args.top_k,
                allowed_scopes=allowed,
                profile=getattr(args, "retrieval_profile", "baseline"),
                max_context_tokens=getattr(args, "max_context_tokens", 1200),
                view=getattr(args, "view", "memory"),
            )
            response = await client.request("retrieve", data)
        elif command in {"correct", "forget"}:
            data.update(fact_id=args.fact_id, expected_version=args.expected_version)
            if command == "correct":
                data["content"] = args.content
            response = await client.request(command, data)
        elif command == "inspect":
            data["fact_id"] = args.chunk_id
            response = await client.request("inspect", data)
        elif command in {"list", "search", "export", "purge", "feedback-list", "feedback-export"}:
            data["allowed_scopes"] = (
                ["project", "workspace", "global"] if args.scope == "all" else [args.scope]
            )
            if command.startswith("feedback"):
                data["event_type"] = args.event_type
                response = await client.request("feedback_list", data)
                items = response.get("events", [])
            else:
                items = []
                while True:
                    page = await client.request("list", {**data, "limit": 500})
                    if page.get("status") == "degraded":
                        print(json.dumps(page, indent=2))
                        return page
                    items.extend(page.get("facts", []))
                    cursor = page.get("next_cursor")
                    if not cursor or not page.get("facts"):
                        break
                    data["after_id"] = cursor
                text = getattr(args, "query", "").casefold()
                if text:
                    items = [item for item in items if text in item["content"].casefold()]
                older = getattr(args, "older_than_days", 0)
                if older:
                    cutoff = datetime.now(timezone.utc) - timedelta(days=older)
                    items = [
                        item
                        for item in items
                        if datetime.fromisoformat(item["updated_at"]) < cutoff
                    ]
                items.sort(
                    key=lambda item: item.get(getattr(args, "sort_by", "created_at"), ""),
                    reverse=True,
                )
            if args.limit > 0:
                items = items[: args.limit]
            response = {"schema_version": 1, "items": items, "count": len(items)}
            if command == "purge":
                if not args.dry_run and not args.yes:
                    raise ValueError("Purge requires --yes or --dry-run")
                if args.dry_run:
                    response["dry_run"] = True
                else:
                    results = [
                        await client.request(
                            "forget",
                            {
                                "context": context.model_dump(),
                                "fact_id": item["id"],
                                "expected_version": item["version"],
                            },
                        )
                        for item in items
                    ]
                    response = {
                        "schema_version": 1,
                        "results": results,
                        "forgotten": sum(result.get("forgotten") is True for result in results),
                    }
            elif command.endswith("export") and args.output:
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                rendered = (
                    "\n".join(json.dumps(item) for item in items) + "\n"
                    if args.format == "jsonl"
                    else json.dumps(response, indent=2)
                )
                with output.open("x", encoding="utf-8") as handle:
                    from .service import _protect_private_path

                    _protect_private_path(output)
                    handle.write(rendered)
                response = {
                    "schema_version": 1,
                    "count": len(items),
                    "output": str(output.resolve()),
                }
        elif command == "feedback":
            data.update(
                event_type=args.event_type,
                query=args.query,
                notes=args.notes,
                chunk_ids=args.chunk_ids,
                scope=args.scope,
            )
            response = await client.request("feedback", data)
        else:
            raise ValueError(f"Unsupported managed command: {command}")
    print(json.dumps(response, indent=2))
    return response
