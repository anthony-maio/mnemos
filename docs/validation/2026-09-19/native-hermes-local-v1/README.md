# Native Hermes local attempts: scope propagation remains open

Both recorded attempts failed their acceptance checks. They use the verified
Mnemos `de6cdcf1c05fdad1bb19ed8f52638eff5ed67ae4` wheel, installed Hermes
`14a346345486418aa85ed6526549412ee2665ade`, local Ollama 0.34.1, and the model
digests in each report. The fixture uses temporary homes, registered projects,
SQLite, and a Windows bridge. Live host configuration was not changed.

Qwen3.5:9b loaded at 65,536 tokens with `size_vram == size == 8,044,454,869`
bytes. Its first load took 79.037 seconds, outside the native turn timing.
The two native CLI processes finished in 18.565 and 17.049 seconds. This
supersedes the earlier observation that no local Hermes turn had completed;
it does not establish safe native Mnemos integration.

Attempt 1 omitted an explicit CLI `--in` and did not process the queued seed
jobs. Provider initialization failed because Hermes supplied no `cwd`;
prefetch returned empty context and the model answered `unknown`. There were
no durable facts. Two raw Hermes events persisted despite failed initialization,
exposing an unintended fallback from an empty adapter directory to process cwd.

Attempt 2 supplied `--in`, processed the seeds, and verified their scoped
retrieval before launching Hermes. Hermes still omitted the provider's `cwd`.
It retrieved the project label and answered correctly only through that same
unsafe fallback. Two user/assistant events persisted in the expected fixture
scope, and the foreign project's canary appeared in neither prefetch nor the
answer. These observations cannot close the session authority gate.

The installed host changes process cwd for `--in`, but its CLI agent constructor
does not pass `cwd` to `AIAgent`. The provider initialization kwargs only include
it when `agent.session_cwd` is set. Relevant installed source locations are
`hermes_cli/main.py`, `hermes_cli/cli_agent_setup_mixin.py`, and
`agent/agent_init.py`. Reading an ambient process directory in the provider
would conceal this propagation gap.

The current Mnemos repair clears prior context and cached recall on initialize
and rejects subsequent operations without a validated session directory. It
also stops treating built-in Hermes remove/replace notifications as fresh
capture. Those categories are not Mnemos fact IDs. Pending lifecycle diagnostics
are bounded, content-free, and in-memory only; automatic cross-store correction
and forgetting remain unsupported without an exact mapping.

Both traces contain two accepted and two failed capture callback returns.
SQLite contains two native raw events; this does not prove every callback
succeeded or that extraction completed. The v2 model observer also undercounts
tools/messages when empty top-level lists coexist with populated extra-body
lists, so its effective-context check fails. Preserve the reports as recorded.
A successor needs explicit host scope propagation, accurate request observation,
and bridge response evidence before claiming a native pass.

`evidence-index.json` hashes 32 copied artifacts, including immutable helper
versions and post-run database observations. It excludes this explanatory file.
Runtime credentials and live memory databases are not included.
