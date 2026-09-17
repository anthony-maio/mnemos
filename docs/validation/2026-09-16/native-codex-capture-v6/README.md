# Native Codex capture v6

This is the second repaired native Codex check. It ran Codex CLI 0.146.0 against
an installed d17 Mnemos wheel, a disposable project, a private SQLite service,
and simple embeddings with cognition unavailable. The hook used the manual
PowerShell wrapper from the probe. It did not exercise the later production
renderer package. The report is preserved at
`report.json`; this folder keeps the exact runner, a short observation record,
and the hash index.

Three native hook calls returned code 0: `SessionStart`, `UserPromptSubmit`,
and `Stop`. The user and assistant capture rows survived a service restart, and
the prompt hook returned the seeded SQLite context. Codex answered `The project
uses SQLite.` for the seeded prompt, so this run records the seeded recall
criteria as observed.

This run does not measure model quality, prove that the
answer came from the memory context, or establish native Codex performance. The
fixture used unavailable cognition and simple embeddings. Reported token usage
is retained as host telemetry, while the model name is unreported. Hook timings
are evidence for this run, not a performance gate.

The private fixture is retained for review. The copied auth file and runtime
descriptor were removed, and no process owned by the fixture remained at the
cleanup check. No live Codex configuration, host memory, or Ollama service was
changed.
