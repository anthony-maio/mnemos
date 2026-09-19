"""Observe a disposable Hermes CLI without replacing its model or tool results."""

from __future__ import annotations

from functools import wraps
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any
import zipfile


def main(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(cfg["phase_root"])
    wheel = Path(cfg["wheel"])
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != cfg["wheel_sha256"]:
        raise ValueError("Wheel hash changed")
    preserved = {
        key: value
        for key, value in os.environ.items()
        if key in {"HOME", "USER", "LOGNAME", "PATH", "LANG", "WSL_INTEROP", "WSL_DISTRO_NAME"}
    }
    os.environ.clear()
    os.environ.update(preserved)
    os.environ.update(cfg["env"])
    sys.dont_write_bytecode = True
    extracted = root / "wheel"
    extracted.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(extracted)
    sys.path[:0] = [str(extracted), cfg["hermes_checkout"]]
    os.chdir(cfg["project"])
    lock = threading.Lock()

    def record(operation: str, **fields: Any) -> None:
        with lock:
            with (root / "trace.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"operation": operation, **fields}, default=str) + "\n")

    bridge = importlib.import_module("mnemos.bridge")
    original_request = bridge.BridgeClient.request

    @wraps(original_request)
    async def request(self: Any, operation: str, payload: Any = None, **kwargs: Any) -> Any:
        result = await original_request(self, operation, payload, **kwargs)
        # This probe's isolated inputs contain only synthetic fixture content.
        # Runtime descriptors and their authentication tokens are never logged.
        record("bridge_response", request_operation=operation, payload=payload, result=result)
        return result

    bridge.BridgeClient.request = request
    adapter = importlib.import_module("mnemos.adapters.hermes")
    manager_module = importlib.import_module("agent.memory_manager")
    provider_type = adapter.MnemosHermesProvider
    manager_type = manager_module.MemoryManager
    original_tool = manager_type.handle_tool_call

    @wraps(original_tool)
    def handle_tool(self: Any, name: str, args: dict[str, Any], **kwargs: Any) -> Any:
        result = original_tool(self, name, args, **kwargs)
        try:
            parsed = json.loads(result)
        except (TypeError, ValueError):
            parsed = result
        record("tool_result", name=name, args=args, result=parsed)
        return result

    manager_type.handle_tool_call = handle_tool

    def observe(name: str) -> None:
        original = getattr(provider_type, name)

        @wraps(original)
        def observed(self: Any, *args: Any, **kwargs: Any) -> Any:
            started = time.monotonic()
            if name == "sync_turn" and kwargs.get("messages"):
                record("transcript", session_id=self.session_id, messages=kwargs["messages"])
            try:
                result = original(self, *args, **kwargs)
            except BaseException as exc:
                record("provider_error", method=name, error_type=type(exc).__name__)
                raise
            record(
                "provider_callback",
                method=name,
                elapsed_ms=(time.monotonic() - started) * 1000,
                session_id=self.session_id,
                cwd=self.cwd,
                hermes_home=self.hermes_home,
                result=result if isinstance(result, (str, bool, type(None))) else None,
            )
            return result

        setattr(provider_type, name, observed)

    for method in ("initialize", "prefetch", "sync_turn", "on_session_end", "shutdown", "_flush"):
        observe(method)

    loop: Any = importlib.import_module("agent.conversation_loop")
    original_call = loop.perform_api_call
    response_number = 0

    @wraps(original_call)
    def model_call(*args: Any, **kwargs: Any) -> Any:
        nonlocal response_number
        verdict = original_call(*args, **kwargs)
        response_number += 1
        response = verdict.response
        calls = []
        content = None
        choices = getattr(response, "choices", None)
        if choices:
            message = choices[0].message
            content = message.content
            for call in message.tool_calls or []:
                calls.append(
                    {
                        "id": f"response-{response_number}:{call.id}" if call.id else "",
                        "provider_call_id": call.id,
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                )
        record("model_response", tool_calls=calls, content=content, action=verdict.action)
        return verdict

    loop.perform_api_call = model_call
    sdk = importlib.import_module("openai.resources.chat.completions")
    original_create = sdk.Completions.create

    @wraps(original_create)
    def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        extra = kwargs.get("extra_body") or {}
        messages = kwargs.get("messages") or extra.get("messages") or []
        serialized = json.dumps(messages, default=str)
        record(
            "model_request",
            model=kwargs.get("model"),
            reasoning_effort=kwargs.get("reasoning_effort"),
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            max_completion_tokens=kwargs.get("max_completion_tokens"),
            options=extra.get("options"),
            message_count=len(messages),
            tools=[
                row.get("function", {}).get("name")
                for row in kwargs.get("tools") or extra.get("tools") or []
            ],
            contains_old_label=cfg["old_label"] in serialized,
            contains_new_label=cfg["new_label"] in serialized,
            contains_foreign_label=cfg["foreign_label"] in serialized,
        )
        return original_create(self, *args, **kwargs)

    sdk.Completions.create = create
    cli = importlib.import_module("hermes_cli.main")
    origins = {
        name: importlib.import_module(name).__file__
        for name in (
            "hermes_cli.main",
            "hermes_cli.cli_agent_setup_mixin",
            "agent.conversation_loop",
            "agent.memory_manager",
        )
    }
    if not all(str(path).startswith(cfg["hermes_checkout"] + "/") for path in origins.values()):
        raise ValueError("Hermes import escaped the isolated checkout")
    if not str(adapter.__file__).startswith(str(extracted) + "/"):
        raise ValueError("Mnemos import escaped the verified wheel")
    title = importlib.import_module("agent.title_generator")
    if title._model_title_upgrade_enabled():
        raise ValueError("Disposable Hermes title model upgrade remains enabled")
    record(
        "bootstrap",
        pid=os.getpid(),
        origins=origins,
        adapter_origin=adapter.__file__,
        title_model_upgrade_enabled=False,
    )
    mode = cfg["mode"]
    if mode in {"preflight", "replay"}:
        manager = manager_type()
        discovery = importlib.import_module("plugins.memory")
        provider = discovery.load_memory_provider("mnemos")
        if not isinstance(provider, provider_type) or not provider.is_available():
            raise ValueError("Hermes could not discover the verified Mnemos provider")
        manager.add_provider(provider)
        manager.initialize_all(
            cfg["session_id"], cwd=cfg["project"], hermes_home=cfg["env"]["HERMES_HOME"]
        )
        if provider._context is None or provider.hermes_home != cfg["env"]["HERMES_HOME"]:
            raise ValueError("Provider failed to resolve the disposable project and identity home")
        try:
            if mode == "replay":
                manager.handle_tool_call("mnemos_remember", cfg["remember_args"])
                manager.on_session_end(cfg["messages"])
                if not manager.flush_pending(timeout=15) or not provider._flush(timeout=15):
                    raise ValueError("Native source replay did not flush")
            record(
                mode + "_complete",
                tool_names=sorted(manager.get_all_tool_names()),
                discovery="plugins.memory.load_memory_provider",
                hermes_home=provider.hermes_home,
            )
        finally:
            manager.shutdown_all()
        return
    if mode != "model":
        raise ValueError("Unknown native probe mode")
    sys.argv = [
        "hermes",
        "chat",
        "--query-file",
        str(root / "query.txt"),
        "--oneshot",
        "--quiet",
        "--in",
        cfg["project"],
        "--model",
        cfg["model"],
        "--provider",
        "custom",
        "--reasoning",
        "none",
        "--toolsets",
        "memory",
        "--max-turns",
        "8",
        "--run-budget",
        "180",
    ]
    cli.main()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
