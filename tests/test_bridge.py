from __future__ import annotations

import json
import asyncio
import os
import sys
from pathlib import Path

import pytest

from mnemos.adapters.hermes import MnemosHermesProvider
from mnemos.bridge import BridgeClient, parse_bridge_command
from mnemos.scopes import ScopeRegistry
from tests.test_cross_host_contract import _start_service


def test_bridge_command_requires_json_argv() -> None:
    assert parse_bridge_command('["python", "-m", "mnemos.bridge"]')[0] == "python"
    with pytest.raises(ValueError, match="JSON argv"):
        parse_bridge_command("python -m mnemos.bridge")
    with pytest.raises(ValueError, match="non-empty"):
        parse_bridge_command("[]")


@pytest.mark.asyncio
async def test_bridge_client_roundtrip_uses_one_json_request(tmp_path: Path) -> None:
    service = _start_service(tmp_path, ScopeRegistry())
    descriptor = service.descriptor_path
    client = BridgeClient(
        [
            sys.executable,
            "-m",
            "mnemos.bridge",
            "--descriptor",
            str(descriptor),
            "--deadline",
            "3",
        ],
        deadline=5,
    )

    try:
        result = await client.request("health", {})
        assert result["status"] == "running"
    finally:
        service.stop()


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform != "win32" or not os.environ.get("MNEMOS_TEST_HERMES_PYTHON"),
    reason="Set MNEMOS_TEST_HERMES_PYTHON to the installed WSL Hermes Python on Windows",
)
async def test_real_wsl_bridge_invokes_windows_client(tmp_path: Path) -> None:
    """Exercise the configured WSL Python bridge against a temporary service."""
    wsl_python = os.environ["MNEMOS_TEST_HERMES_PYTHON"]
    project = tmp_path / "project"
    project.mkdir()
    registry = ScopeRegistry()
    profile = "wsl-bridge-profile"
    registry.register_project(project, user_profile_id=profile, enabled=True)
    from tests.test_cross_host_contract import _start_service

    service = _start_service(tmp_path, registry)
    descriptor = str(service.descriptor_path)
    source_root = str(Path(__file__).resolve().parents[1]).replace("\\", "/")
    source_root = "/mnt/" + source_root[0].lower() + source_root[2:]
    windows_python = str(Path(sys.executable)).replace("\\", "/")
    windows_python = "/mnt/" + windows_python[0].lower() + windows_python[2:]
    context = registry.resolve(
        project, user_profile_id=profile, session_id="wsl-session", host="hermes"
    ).model_dump(mode="json")
    script = f"""
import asyncio, json, sys
sys.path.insert(0, {source_root!r})
from mnemos.bridge import BridgeClient
async def main():
    client = BridgeClient([{windows_python!r}, '-m', 'mnemos.bridge', '--descriptor', {descriptor!r}], deadline=10)
    context = {json.dumps(context)!r}
    context = json.loads(context)
    first = await client.request('remember', {{'context': context, 'event_id': 'wsl-event', 'content': 'WSL bridge continuity fact.', 'scope': 'project'}}, queue_on_failure=False)
    second = await client.request('maintenance', {{'context': context}}, queue_on_failure=False)
    print(json.dumps({{'remember': first, 'maintenance': second}}))
asyncio.run(main())
"""
    try:
        process = await asyncio.create_subprocess_exec(
            "wsl.exe",
            "-e",
            wsl_python,
            "-c",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
        assert process.returncode == 0, stderr.decode(errors="replace")
        result = json.loads(stdout.decode())
        assert result["remember"]["status"] == "committed"
        assert result["maintenance"]["completed"] >= 1
    finally:
        service.stop()


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform != "win32" or not os.environ.get("MNEMOS_TEST_HERMES_PYTHON"),
    reason="Set MNEMOS_TEST_HERMES_PYTHON to the installed WSL Hermes Python on Windows",
)
async def test_installed_hermes_provider_contract_in_isolated_wsl_process() -> None:
    """Load the installed Hermes base class without starting a Hermes session."""
    wsl_python = os.environ["MNEMOS_TEST_HERMES_PYTHON"]
    source_root = str(Path(__file__).resolve().parents[1]).replace("\\", "/")
    source_root = "/mnt/" + source_root[0].lower() + source_root[2:]
    script = f"""
import asyncio, json, pathlib, sys, tempfile
sys.path.insert(0, {source_root!r})
from agent.memory_provider import MemoryProvider
from mnemos.adapters.hermes import MnemosHermesProvider
from mnemos.scopes import ScopeRegistry
class Client:
    def __init__(self): self.captures = 0
    async def capture(self, payload): self.captures += 1; return {{'status': 'committed', 'committed': True}}
    async def retrieve(self, payload): return {{'facts': [{{'content': 'isolated Hermes context'}}]}}
    async def request(self, operation, payload, **kwargs): return {{'status': 'committed'}}
async def main():
    project = pathlib.Path(tempfile.mkdtemp(prefix='mnemos-hermes-contract-'))
    registry = ScopeRegistry()
    registry.register_project(project, user_profile_id='hermes-isolated', enabled=True)
    client = Client()
    provider = MnemosHermesProvider(client=client, registry=registry, user_profile_id='hermes-isolated')
    provider.initialize('isolated-session', cwd=str(project))
    recalled = provider.prefetch('what is the context?')
    provider.sync_turn('A user fact', 'An assistant response', session_id='isolated-session')
    checkpoint = provider.on_pre_compress([{{'role':'user','content':'Checkpoint fact'}}], require_checkpoint=True)
    provider.shutdown()
    print(json.dumps({{'base_class': isinstance(provider, MemoryProvider), 'transport': provider.transport, 'recalled': recalled, 'checkpoint': checkpoint, 'captures': client.captures}}))
asyncio.run(main())
"""
    process = await asyncio.create_subprocess_exec(
        "wsl.exe",
        "-e",
        wsl_python,
        "-c",
        script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
    assert process.returncode == 0, stderr.decode(errors="replace")
    result = json.loads(stdout.decode())
    assert result["base_class"] is True
    assert result["transport"] == "injected"
    assert "isolated Hermes context" in result["recalled"]
    assert result["checkpoint"]
    assert result["captures"] >= 2


@pytest.mark.asyncio
async def test_bridge_client_bounds_response_and_timeout() -> None:
    noisy = BridgeClient(
        [sys.executable, "-c", "import sys; sys.stdin.read(); print('x' * 200)"],
        max_response_bytes=32,
    )
    assert (await noisy.request("health", {}))["status"] == "degraded"
    assert "size limit" in (await noisy.request("health", {}))["error"]

    slow = BridgeClient(
        [sys.executable, "-c", "import sys,time; sys.stdin.read(); time.sleep(2)"],
        deadline=0.1,
    )
    timed = await slow.request("health", {})
    assert timed["status"] == "degraded"
    assert "timed out" in timed["error"]


def test_hermes_uses_bridge_only_when_explicitly_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = json.dumps([sys.executable, "-m", "mnemos.bridge", "--descriptor", "runtime.json"])
    monkeypatch.setenv("MNEMOS_BRIDGE_COMMAND", command)
    provider = MnemosHermesProvider(
        registry=ScopeRegistry(profiles=["alice"]), user_profile_id="alice"
    )

    assert provider.transport == "explicit_bridge"
    assert provider.client.command[:3] == (sys.executable, "-m", "mnemos.bridge")
