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

## Changed retry hypothesis: September 17, 2026

Read-only inspection of installed Hermes 0.21.3 at source `36842e63` found
the per-run `--reasoning none` option. Its custom provider maps disabled
reasoning to top-level `reasoning_effort="none"` and, for an Ollama endpoint,
`think=false`. The inspected implementation is
`plugins/model-providers/custom/__init__.py`, lines 39-51. The top-level
reasoning field matters for the OpenAI-compatible endpoint; the provider
source explicitly notes that this endpoint ignores the extra-body think field.

The preserved earlier attempt does not record disabled reasoning or the exact
temporary configuration. A new isolated attempt can therefore test a concrete
change: explicit custom provider, `qwen3.5:9b`, reasoning disabled, and both
`context_length` and `ollama_num_ctx` set to 65536 in a disposable home.
It must retain normal approvals and record the effective request options,
provider callbacks, model digest, and a completed turn before claiming success.
This retry has not run. The active serialized Ollama evaluation takes precedence;
the live Hermes configuration and gateway have not been changed.
