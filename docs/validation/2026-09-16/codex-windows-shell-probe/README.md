# Codex Windows shell boundary probe

This was an offline check using the development interpreter and source tree
through `PYTHONPATH`. It started one disposable loopback service with SQLite storage,
simple deterministic embeddings, and `UnavailableLLMProvider`. It did not
start Codex, read credentials, contact a host, or call a model.

The same synthetic `UserPromptSubmit` payload was sent through two paths:

- direct `python -m mnemos.adapters.codex`
- the current `_codex_hook_command` string wrapped as the tagged Windows
  `cmd.exe /d /c "<hook command>"` form

Both subprocesses exited zero with empty stderr. The two stored user episodes
were the seed and one shared prompt identity. Since both invocations reused
that identity, this result cannot establish that each path captured a separate
event. Neither path returned context; the seed had not been consolidated. The
empty stdout and successful exits do not establish seeded recall. These limits
are addressed by the [separate v2 probe](../codex-windows-shell-probe-v2/README.md).

The result records the generated command string, the exact raw `cmd.exe`
command line, source hashes, integration environment keys, return codes, and
sanitized stderr. Temporary service state was removed after the run. The
generated command includes only allowlisted paths and environment values; no
credential or model output was present.

The shell construction is based on the tagged Codex command runner, which
uses Windows `raw_arg` quoting for the hook command:

- [Codex command runner at rust-v0.146.0](https://raw.githubusercontent.com/openai/codex/rust-v0.146.0/codex-rs/hooks/src/engine/command_runner.rs)
- [Codex hook discovery at rust-v0.146.0](https://raw.githubusercontent.com/openai/codex/rust-v0.146.0/codex-rs/hooks/src/engine/discovery.rs)

See [result.json](result.json) for the captured evidence. This probe does
not establish native Codex lifecycle dispatch or cancellation behavior. The
corrected wheel-isolated follow-up is documented in [v2](../codex-windows-shell-probe-v2/README.md).
V1 used the editable worktree import, did not consolidate its seed, reused one
turn identity, and removed its fixture. Its result is retained as historical
evidence with those limits.
