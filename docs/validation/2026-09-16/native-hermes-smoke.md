# Native Hermes WSL smoke

Date: 2026-09-16

This bounded check used the installed Hermes Agent WSL environment at
`/home/anthony/.hermes/hermes-agent`, version `0.21.3`. The installed
`agent.memory_provider.MemoryProvider` class imported successfully and exposed
the lifecycle methods used by Mnemos: initialize, prefetch, sync_turn,
on_pre_compress, on_session_end, and shutdown.

The run created a temporary Hermes home and user memory plugin outside the
repository. The plugin registered `MnemosHermesProvider` through Hermes'
actual `ctx.register_memory_provider(...)` discovery path. A temporary
authenticated Windows Mnemos service seeded the scoped fact
`The project uses SQLite for durable memory storage.` and completed one
maintenance job. WSL used the explicit JSON `MNEMOS_BRIDGE_COMMAND` to invoke
the Windows Python bridge. No live Hermes config, memory, credentials, or
network listener was changed.

The first local model, `anthony-maio/eve-2-moe-it-272m:q4-k-m`, was rejected
before a turn because Hermes requires a context window of at least 64K tokens;
the Ollama model metadata reports 2,048. `gemma3:4b` has a sufficient reported
window, but Ollama rejected Hermes' tool-bearing request because that model
does not support tools. The installed tool-capable `qwen3.5:9b` was then tried
with a disposable 64K context setting and a 240-second deadline. It emitted no
stdout or stderr and timed out before the runner could observe a completed
turn or provider callback.

The sanitized machine-readable record is
[native-hermes-smoke.json](native-hermes-smoke.json).

Acceptance state:

- Installed Hermes `MemoryProvider` API: OBSERVED.
- Disposable provider registration path: prepared through the real plugin loader.
- Explicit WSL-to-Windows bridge and scoped service seed: OBSERVED.
- Native Hermes agent turn: NOT OBSERVED.
- Native Hermes prefetch, sync_turn, and durable capture: NOT OBSERVED.
- Native Hermes recall after a real turn: NOT OBSERVED.

The remaining gate needs a local Ollama model that both supports Hermes tool
requests and returns within the host's bounded run budget, or an explicit
Hermes configuration for a qualifying model. Mnemos should continue to label
Hermes native CLI integration as unproven and retain the manual or MCP path.
