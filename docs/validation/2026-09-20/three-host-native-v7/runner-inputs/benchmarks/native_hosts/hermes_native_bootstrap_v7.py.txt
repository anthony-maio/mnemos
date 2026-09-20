"""Keep the v6 native invocation and verify loaded module origins at exit."""

from __future__ import annotations

from importlib.machinery import SourceFileLoader
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping


def exit_origins(cfg: Mapping[str, Any], modules: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "hermes_cli.main",
        "hermes_cli.cli_agent_setup_mixin",
        "agent.conversation_loop",
        "agent.memory_manager",
    )
    origins = {name: getattr(modules.get(name), "__file__", None) for name in names}
    checkout = PurePosixPath(cfg["hermes_checkout"])
    if not all(
        isinstance(path, str) and PurePosixPath(path).is_relative_to(checkout)
        for path in origins.values()
    ):
        raise ValueError("Hermes import escaped the isolated checkout at exit")
    adapter = getattr(modules.get("mnemos.adapters.hermes"), "__file__", None)
    wheel_root = PurePosixPath(cfg["phase_root"]) / "wheel"
    if not isinstance(adapter, str) or not PurePosixPath(adapter).is_relative_to(wheel_root):
        raise ValueError("Mnemos import escaped the verified wheel at exit")
    return {"operation": "bootstrap_exit", "origins": origins, "adapter_origin": adapter}


def main(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    loader = SourceFileLoader(
        "mnemos_bootstrap_v6", str(Path(__file__).with_name("hermes_native_bootstrap_v6.py.txt"))
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the unchanged v6 bootstrap")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.main(config_path)
    finally:
        row = exit_origins(cfg, sys.modules)
        with (Path(cfg["phase_root"]) / "trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
