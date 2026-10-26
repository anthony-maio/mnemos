"""Install a built wheel outside the checkout and run restart/recall through its CLIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from email.parser import BytesParser
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def clean_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(("MNEMOS_", "GIT_"))
        and key.upper() not in {"PYTHONPATH", "PYTHONHOME"}
    }


def wheel_version(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("Expected exactly one wheel metadata file")
        parsed = BytesParser().parsebytes(archive.read(metadata[0]))
    if parsed["Name"] != "mnemos-memory":
        raise ValueError("Expected a mnemos-memory wheel")
    return str(parsed["Version"])


def run(argv: list[str], *, cwd: Path, env: dict[str, str], timeout: int = 1200) -> str:
    result = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(f"Installed artifact check failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument(
        "--work-dir", type=Path, help="External parent for a disposable consumer venv"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--release-tag", help="Require the release tag to match wheel/runtime version"
    )
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    version = wheel_version(wheel)
    if args.release_tag and args.release_tag != f"v{version}":
        raise ValueError("Release tag and wheel version do not match")
    parent = args.work_dir.resolve() if args.work_dir else Path(tempfile.gettempdir()).resolve()
    if parent == ROOT or ROOT in parent.parents:
        raise ValueError("The consumer environment must be outside the source checkout")
    parent.mkdir(parents=True, exist_ok=True)
    env = clean_env()
    env.update(MNEMOS_AUTO_START="0", PYTHONUTF8="1")
    with tempfile.TemporaryDirectory(prefix="mnemos-installed-", dir=parent) as directory:
        workspace = Path(directory)
        venv.EnvBuilder(with_pip=True).create(workspace / "venv")
        scripts = workspace / "venv" / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        run([str(python), "-m", "pip", "install", f"{wheel}[mcp]"], cwd=workspace, env=env)
        check = workspace / "check-install.py"
        check.write_text(
            "import importlib.metadata as md, json, pathlib, sys\n"
            "import mnemos\n"
            "from mcp.server.fastmcp import FastMCP\n"
            "from importlib.resources import files\n"
            "from mnemos.benchmark import _resolve_dataset_pack, load_documents\n"
            "package = pathlib.Path(mnemos.__file__).resolve()\n"
            "assert pathlib.Path(sys.prefix).resolve() in package.parents\n"
            f"assert md.version('mnemos-memory') == mnemos.__version__ == {version!r}\n"
            "assert int(md.version('mcp').split('.')[0]) == 1\n"
            "assert files('mnemos').joinpath('py.typed').is_file()\n"
            "scripts = {e.name: e.value for e in md.distribution('mnemos-memory').entry_points "
            "if e.group == 'console_scripts'}\n"
            "assert {'mnemos', 'mnemos-cli', 'mnemos-mcp', 'mnemos-benchmark'} <= scripts.keys()\n"
            "provider = list(md.entry_points(group='hermes_agent.memory_providers', name='mnemos'))\n"
            "assert len(provider) == 1 and callable(provider[0].load())\n"
            "datasets = _resolve_dataset_pack('claim-driving')\n"
            "assert len(datasets) == 3\n"
            "dataset_counts = {}\n"
            "for name, path in datasets:\n"
            "    assert package.parent in path.resolve().parents\n"
            "    dataset_counts[name] = len(load_documents(path))\n"
            "    assert dataset_counts[name] > 0\n"
            "print(json.dumps({'version': md.version('mnemos-memory'), "
            "'mcp_version': md.version('mcp'), 'fastembed_version': md.version('fastembed'), "
            "'console_scripts': scripts, 'hermes_entrypoint': provider[0].value, "
            "'bundled_benchmark_documents': dataset_counts}))\n",
            encoding="utf-8",
        )
        installation = json.loads(run([str(python), "-I", str(check)], cwd=workspace, env=env))
        for name in ("mnemos", "mnemos-cli", "mnemos-benchmark"):
            executable = scripts / (name + (".exe" if os.name == "nt" else ""))
            run([str(executable), "--help"], cwd=workspace, env=env)
        example = workspace / "restart_recall.py"
        shutil.copyfile(ROOT / "examples" / "restart_recall.py", example)
        example_receipt = workspace / "restart-recall.json"
        print(
            "Installed wheel verified; running real FastEmbed, CLI and MCP restart/recall.",
            flush=True,
        )
        run(
            [
                str(python),
                "-I",
                str(example),
                "--work-dir",
                str(workspace),
                "--output",
                str(example_receipt),
            ],
            cwd=workspace,
            env=env,
        )
        receipt: dict[str, Any] = {
            "status": "passed",
            "wheel": wheel.name,
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "installation": installation,
            "example": json.loads(example_receipt.read_text(encoding="utf-8")),
        }
    rendered = json.dumps(receipt, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
