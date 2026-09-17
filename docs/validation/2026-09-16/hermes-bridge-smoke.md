# Hermes bridge smoke

Date: 2026-09-16

This is an ephemeral transport and provider contract record. It does not
change the live Hermes configuration, install a package, or start a real host
session.

## Windows service through WSL

The test started one temporary authenticated loopback `MnemosService` with a
temporary SQLite store and enabled project registry. A WSL Python process used
`BridgeClient` with an explicit `create_subprocess_exec` argv targeting the
Windows worktree Python executable and `python -m mnemos.bridge`. The bridge
then used the Windows `MnemosClient` and the existing runtime descriptor.

The WSL process sent an explicit `remember` request and a `maintenance` request
with a Hermes scope context. Both responses were JSON. The remember operation
was committed and maintenance processed at least one job. The service and all
temporary files were stopped or discarded after the test.

Command:

```text
.venv\Scripts\python.exe -m pytest -q tests/test_bridge.py::test_real_wsl_bridge_invokes_windows_client
```

Result: `1 passed`.

## Installed Hermes provider API

The isolated WSL process used Hermes Agent `v0.21.3` with upstream revision
`97962358` from `/home/anthony/.hermes/hermes-agent`. It loaded the installed
`agent.memory_provider.MemoryProvider` base class and Mnemos' provider with an
injected synthetic client and temporary registry. It exercised `initialize`,
`prefetch`, `sync_turn`, `on_pre_compress(require_checkpoint=True)`, and
`shutdown`. The provider was an instance of the installed base class, recall
was returned, and the checkpoint was accepted.

Command:

```text
.venv\Scripts\python.exe -m pytest -q tests/test_bridge.py::test_installed_hermes_provider_contract_in_isolated_wsl_process
```

Result: `1 passed`.

## Evidence boundary

This record proves the explicit WSL to Windows bridge transport and the
installed Hermes provider interface in an isolated synthetic process. It does
not prove a real Hermes CLI conversation, a real Codex or Claude Code session,
provider extraction quality, model readiness, or native end-to-end host
continuity. Those remain NOT RUN.
