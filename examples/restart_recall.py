"""Verify explicit memory across CLI, service and MCP process restarts.

Install mnemos-memory[mcp], then run this file with that environment's Python.
The first run downloads FastEmbed's local embedding model. No hosted LLM or
judgment service is used. This is a persistence and integration example, not
an evaluation of automatic extraction or retrieval quality.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

FACT = "The sample notebook project keeps release checklists in docs/release.md."
QUERY = "Where does the sample notebook project keep its release checklist?"
PROFILE = "restart-recall-example"


def isolated_env(root: Path) -> dict[str, str]:
    """Select only this example's config, service, registry and database."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(("MNEMOS_", "GIT_"))
        and key.upper() not in {"PYTHONPATH", "PYTHONHOME"}
    }
    env.update(
        MNEMOS_CONFIG_PATH=str(root / "mnemos.toml"),
        MNEMOS_RUNTIME_DIR=str(root / "runtime"),
        MNEMOS_RUNTIME_DESCRIPTOR=str(root / "runtime" / "service.json"),
        MNEMOS_SCOPE_REGISTRY=str(root / "scopes.json"),
        MNEMOS_SQLITE_PATH=str(root / "memory.sqlite"),
        MNEMOS_STORE_TYPE="sqlite",
        MNEMOS_USER_PROFILE_ID=PROFILE,
        MNEMOS_AUTO_START="0",
        MNEMOS_EMBEDDING_PROVIDER="fastembed",
        MNEMOS_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5",
        PYTHONUTF8="1",
    )
    return env


def entrypoint(name: str) -> str:
    script = Path(sys.executable).parent / (name + (".exe" if os.name == "nt" else ""))
    if not script.is_file():
        raise RuntimeError(f"Install mnemos-memory[mcp] in this Python environment: missing {name}")
    return str(script)


def run_process(argv: list[str], *, env: dict[str, str], cwd: Path, timeout: int = 120) -> str:
    result = subprocess.run(
        argv, env=env, cwd=cwd, capture_output=True, text=True, encoding="utf-8", timeout=timeout
    )
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {argv[0]}\n{result.stderr}")
    return result.stdout


def cli(args: list[str], *, env: dict[str, str], cwd: Path) -> dict[str, Any]:
    response = json.loads(run_process([entrypoint("mnemos-cli"), *args], env=env, cwd=cwd))
    if not isinstance(response, dict) or response.get("status") in {"degraded", "rejected"}:
        raise RuntimeError(f"CLI operation failed: {response}")
    return response


def assert_recalled(payload: dict[str, Any], fact_id: str, project_id: str) -> None:
    facts = payload.get("facts", [])
    if len(facts) != 1 or facts[0]["id"] != fact_id or facts[0]["content"] != FACT:
        raise AssertionError("Recall did not return the original explicit fact")
    if facts[0]["scope"] != "project" or facts[0]["scope_id"] != project_id:
        raise AssertionError("Recall returned the wrong scope")
    if not facts[0].get("sources") or not facts[0].get("support_available"):
        raise AssertionError("Recall lost its source evidence")


def start_service(root: Path, env: dict[str, str], log: Any) -> subprocess.Popen[bytes]:
    process = subprocess.Popen(
        [sys.executable, "-I", "-m", "mnemos.service"],
        env=env,
        cwd=root,
        stdout=log,
        stderr=log,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    descriptor = Path(env["MNEMOS_RUNTIME_DESCRIPTOR"])
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log.flush()
            log.seek(0)
            details = log.read().decode("utf-8", errors="replace")[-6000:]
            raise RuntimeError(f"The isolated memory service exited during startup:\n{details}")
        if descriptor.is_file():
            return process
        time.sleep(0.1)
    process.terminate()
    process.wait(timeout=30)
    raise TimeoutError("The isolated memory service did not become ready")


def stop_service(process: subprocess.Popen[bytes], env: dict[str, str], cwd: Path) -> None:
    try:
        if process.poll() is None:
            stopped = cli(["service-stop"], env=env, cwd=cwd)
            if stopped.get("status") != "stopped":
                raise RuntimeError("Service shutdown was not confirmed")
        process.wait(timeout=30)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
    if process.returncode != 0 or Path(env["MNEMOS_RUNTIME_DESCRIPTOR"]).exists():
        raise RuntimeError("Service did not exit cleanly")


async def mcp_recall(env: dict[str, str], cwd: Path, fact_id: str, project_id: str) -> int:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(command=entrypoint("mnemos-mcp"), env=env, cwd=str(cwd))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            required = {"mnemos_store", "mnemos_retrieve", "mnemos_inspect", "mnemos_job_status"}
            assert required.issubset({tool.name for tool in tools.tools})

            async def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
                result = await session.call_tool(name, arguments)
                assert not result.isError, result
                return json.loads(
                    "\n".join(item.text for item in result.content if item.type == "text")
                )

            recalled = await call(
                "mnemos_retrieve", {"query": QUERY, "allowed_scopes": ["project"], "top_k": 3}
            )
            assert_recalled(recalled, fact_id, project_id)
            inspected = await call("mnemos_inspect", {"chunk_id": fact_id})
            assert inspected["fact"]["id"] == fact_id
            assert inspected["fact"]["content"] == FACT
            assert inspected["evidence"]
            return len(tools.tools)


def run_example(root: Path) -> dict[str, Any]:
    from importlib.metadata import entry_points, version

    root.mkdir(parents=True, exist_ok=True)
    alpha, beta = root / "notebook", root / "unrelated-project"
    for directory in (root, alpha, beta, root / "runtime"):
        directory.mkdir(exist_ok=True)
        (directory / ".mnemos").mkdir(exist_ok=True)
        (directory / ".mnemos" / "mnemos.toml").write_text("", encoding="utf-8")
    (root / "mnemos.toml").write_text(
        '[judgment]\nprovider = "none"\n'
        "[modules]\nsurprisal = false\nmutable_rag = false\naffective = false\n"
        "sleep = false\nspreading = false\n"
        '[retrieval]\nprofile = "baseline"\naffective_enabled = false\n'
        'spreading_enabled = false\nquery_state_source = "none"\n',
        encoding="utf-8",
    )
    env = isolated_env(root)
    # Work directories may live below another Git checkout. Real, separate
    # repositories prevent both examples from inheriting that parent's identity.
    for project in (alpha, beta):
        run_process(["git", "init", "--quiet"], env=env, cwd=project)
    # Warm the real model before the service's bounded startup window. This may
    # download model files on first use; no memory or host config is modified.
    run_process(
        [
            sys.executable,
            "-I",
            "-c",
            "from mnemos.utils.embeddings import FastembedProvider; "
            "p=FastembedProvider(); assert len(p.embed('release checklist')) == 384",
        ],
        env=env,
        cwd=root,
        timeout=900,
    )
    process: subprocess.Popen[bytes] | None = None
    with (root / "example-service.log").open("w+b") as log:
        try:
            process = start_service(root, env, log)
            registration = cli(
                ["enable-project", "--profile", PROFILE, "--path", str(alpha)], env=env, cwd=alpha
            )
            project_id = registration["project"]["id"]
            health = cli(["stats"], env=env, cwd=alpha)
            assert health["runtime"]["cognition_provider"] == "UnavailableLLMProvider"
            assert health["runtime"]["embedding_provider"] == "FastembedProvider"
            other = cli(
                ["enable-project", "--profile", PROFILE, "--path", str(beta)], env=env, cwd=beta
            )
            assert other["project"]["id"] != project_id
            stored = cli(["store", FACT, "--scope", "project"], env=env, cwd=alpha)
            assert stored["status"] == "committed" and stored["job_id"]
            # The historical CLI name requests one maintenance pass. With every
            # enrichment module disabled, this only finishes the explicit
            # remember job; it does not extract or invent semantic memories.
            cli(["consolidate"], env=env, cwd=alpha)
            deadline = time.monotonic() + 120
            fact_id = ""
            while time.monotonic() < deadline:
                job = cli(["job-status", stored["job_id"]], env=env, cwd=alpha)["job"]
                if job["status"] == "completed":
                    facts = job["result"]["fact_ids"]
                    assert len(facts) == 1
                    fact_id = facts[0]
                    break
                if job["status"] == "failed":
                    raise RuntimeError("Explicit remember job failed")
                time.sleep(0.2)
            assert fact_id, "Explicit remember job did not finish"
            first_pid = process.pid
            stop_service(process, env, alpha)
            process = None

            process = start_service(root, env, log)
            assert process.pid != first_pid
            recalled = cli(["retrieve", QUERY, "--allowed-scopes", "project"], env=env, cwd=alpha)
            assert_recalled(recalled, fact_id, project_id)
            tools_count = asyncio.run(
                asyncio.wait_for(mcp_recall(env, alpha, fact_id, project_id), timeout=90)
            )
            isolated = cli(["retrieve", QUERY, "--allowed-scopes", "project"], env=env, cwd=beta)
            assert isolated["facts"] == [], "A separate project saw the notebook memory"
            hermes = list(entry_points(group="hermes_agent.memory_providers", name="mnemos"))
            assert len(hermes) == 1 and callable(hermes[0].load())
            return {
                "status": "passed",
                "version": version("mnemos-memory"),
                "fact": FACT,
                "query": QUERY,
                "fact_id": fact_id,
                "embedding": "FastEmbed BAAI/bge-small-en-v1.5 (384 dimensions)",
                "experimental_modules": "disabled",
                "hosted_llm_calls": "not required",
                "checks": {
                    "cli_remember": True,
                    "service_restart": True,
                    "cli_recall_same_id": True,
                    "mcp_stdio_initialize": True,
                    "mcp_tools_discovered": tools_count,
                    "mcp_recall_and_inspect": True,
                    "other_project_empty": True,
                    "hermes_entrypoint_loaded": True,
                },
                "limits": "Explicit fact persistence only; no automatic extraction, ranking-quality or native Hermes acceptance claim.",
            }
        finally:
            if process is not None:
                stop_service(process, env, alpha)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the public-safe verification receipt")
    parser.add_argument(
        "--work-dir", type=Path, help="Parent for a new disposable example directory"
    )
    args = parser.parse_args()
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mnemos-restart-", dir=args.work_dir) as directory:
        receipt = run_example(Path(directory))
    receipt["checks"]["service_stopped"] = True
    rendered = json.dumps(receipt, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
