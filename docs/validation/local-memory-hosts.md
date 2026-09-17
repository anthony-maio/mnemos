# Local memory host validation

This record separates adapter and transport checks from real host acceptance. It
describes the unreleased functional-local-memory worktree, updated September 17, 2026.

## Cross-host continuity checkpoint

`tests/test_cross_host_contract.py` starts one authenticated loopback
`MnemosService` with a temporary SQLite database and an in-memory scope
registry. It uses the real `MnemosClient` HTTP transport and three adapter
context types: Codex, Claude Code, and Hermes. Cognition is deliberately
synthetic and disabled for this transport fixture. That makes this a durable
identity and scope test, not a model quality result.

The fixture captures one explicit fact per host, processes each durable job,
corrects the Codex fact through the Claude Code context, and verifies that the
fact keeps its ID while moving to the next version. It stops and restarts the
service, verifies the corrected fact, forgets it through the Hermes context,
and replays the original Codex event. The replay is suppressed. A second
enabled project writes a canary that is absent from project A retrieval.

Run it with:

```text
.venv\Scripts\python.exe -m pytest -q tests/test_cross_host_contract.py
```

This checkpoint passed on the recorded run. It proves authenticated local
service continuity for synthetic explicit-provider fixtures. It does not prove
that any host actually started a session or that a real model generated a
durable fact.

## Installed host evidence

These version checks were read-only:

| Host | Observed installation | Capability evidence | Acceptance state |
| --- | --- | --- | --- |
| Codex CLI | `codex-cli 0.146.0` | Installed source20f generated TOML hooks returned a random canary and captured both roles across restart | Bounded native transport observed; full correction/forget lifecycle pending |
| Codex Desktop | Installed desktop app; no isolated native run | Shared adapter tests only | NOT RUN |
| Claude Code | `2.1.191` | Installed source20f native canary recall and both-role capture survived restart; prompt and Stop identity coverage was degraded | Partial; persistent-transcript attempt timed out. Extraction and full lifecycle matrix remain open |
| Hermes Agent in WSL | `v0.21.3`; initial inspection `97962358`, later wheel discovery `36842e639a178741b70c821e0f4cdd129cd0dfe3` | Installed provider API and authenticated Windows bridge exercised; verified source9e wheel entry point loaded and reported available without service/network calls | Native model conversation did not complete |

The Hermes package is installed under
`/home/anthony/.hermes/hermes-agent` in WSL. The Windows Mnemos environment
does not claim that package as an installed native dependency. The Hermes
adapter reports manual MCP fallback when its provider API is unavailable.

The [native host smoke record](2026-09-16/native-host-smoke.md) preserves the
Claude model-version failure, successful retries, persistent-transcript timeout,
and earlier Codex failures. The later [Codex generated-hook result](2026-09-16/native-codex-generated-v8/README.md)
and [Claude scope result](2026-09-16/native-claude-scope-v4/README.md) preserve
the bounded positive observations. These findings do not replace the complete acceptance
matrix below.

## Safe ephemeral adapter checks

The following commands exercise adapter stdin contracts without changing a
host configuration. They require a temporary running Mnemos descriptor and a
temporary registry whose project is enabled for the selected profile. Set the
environment variables only in the current shell, and clear them afterward.

```powershell
$oldProfile = $env:MNEMOS_USER_PROFILE_ID
$oldRegistry = $env:MNEMOS_SCOPE_REGISTRY
$oldDescriptor = $env:MNEMOS_RUNTIME_DESCRIPTOR
$env:MNEMOS_USER_PROFILE_ID = "host-validation"
$env:MNEMOS_SCOPE_REGISTRY = "C:\Temp\mnemos-host-validation\scopes.json"
$env:MNEMOS_RUNTIME_DESCRIPTOR = "C:\Temp\mnemos-host-validation\runtime.json"
$payload = '{"hook_event_name":"UserPromptSubmit","cwd":"C:\Temp\mnemos-host-validation\project","session_id":"codex-validation","turn_id":"turn-1","prompt":"Use the local test convention."}'
$payload | .venv\Scripts\python.exe -m mnemos.adapters.codex UserPromptSubmit
$env:MNEMOS_USER_PROFILE_ID = $oldProfile
$env:MNEMOS_SCOPE_REGISTRY = $oldRegistry
$env:MNEMOS_RUNTIME_DESCRIPTOR = $oldDescriptor
```

The Claude adapter accepts the same temporary descriptor and registry shape:

```powershell
$payload = '{"hook_event_name":"UserPromptSubmit","cwd":"C:\Temp\mnemos-host-validation\project","session_id":"claude-validation","prompt":"Use the local test convention."}'
$payload | .venv\Scripts\python.exe -m mnemos.adapters.claude_code UserPromptSubmit
```

For Hermes, run the provider contract in its WSL environment with an explicit
descriptor path and profile. The Windows to WSL transport must use an approved
explicit bridge or descriptor. Do not guess a WSL address, expose the service
on a LAN interface, or open SQLite directly across the boundary. Hermes native
activation also requires installing the Mnemos provider into that Hermes
environment; until then, manual MCP is the honest path.

These are adapter checks. A real host gate must record the host version, exact
temporary or operator-selected config, MCP transport, Mnemos descriptor,
profile, project identity, provider configuration, provider observation, and
the capture, retrieval, restart, correction, forget, and outage results.

## Existing validation boundaries

The focused MCP compatibility tests cover capture, maintenance, job status,
inspect, retrieve, feedback, versioned correction, and versioned forget over a
temporary local service. Other transport tests cover bearer authentication,
origin rejection, body limits, one owning asyncio thread, process-lock
startup, restart behavior, and sanitized outbox replay.

The Claude adapter tests cover transcript deltas, role tagging, checkpoints,
stable event IDs, malformed records, partial lines, transcript rotation, scope
resolution, and native context output. Codex and Hermes tests cover optional
imports, injected clients, failure behavior, stable identities, scope
resolution, and lifecycle contracts.

## Open gates

The host continuity checkpoint does not establish extraction precision,
retrieval quality, installed-host reliability, a two-week pilot, or
published-release acceptance. Real-provider preflight and the 10,000-fact
measurement have separate records: [evaluation status](2026-09-16/evaluation-running.md)
and [performance](2026-09-16/performance-10k.md). The held-out evaluation is
still running, and the recorded recall latency fails its gate. The daily-use
pilot and full native cross-host lifecycle remain NOT RUN.
