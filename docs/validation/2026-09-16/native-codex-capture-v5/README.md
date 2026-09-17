# Native Codex capture v5

This is a bounded native Codex 0.146.0 check against a disposable Mnemos
service, project, registry, and SQLite store. The Mnemos fixture used simple
embeddings and unavailable cognition, so this is a host capture and seeded
context check. It makes no extraction-quality claim.

The existing [report](report.json) is preserved byte-for-byte. It records
three wrapper invocations: `SessionStart` timed out, while `UserPromptSubmit`
and `Stop` completed. The prompt hook returned the seeded SQLite context. Both
user and assistant events were captured and survived a service restart.

The overall result remains `not_observed`. Codex answered `The project uses
PostgreSQL.` even though the seeded fact said SQLite. That is a failed native
seeded-recall outcome. The successful wrapper calls establish partial native
dispatch and prompt context output; they do not establish complete
lifecycle capture or correct host recall.

The host stream also contains three error items: two notices that
`--dangerously-bypass-hook-trust` was enabled and one notice that skill
descriptions were shortened to fit the context budget. The reviewed,
sanitized synthetic answer and notices are in [observations.json](observations.json).
Numeric usage is retained there because it was present in the host stream; it
is not a quality or cost acceptance claim.

The exact runner used for the report is preserved in
[runner.py.txt](runner.py.txt). Its SHA-256 matches the report's recorded
runner hash. The evidence index records the report, runner, observation, and
private fixture hashes and cleanup checks. The fixture remains outside this
repository at the path recorded in [evidence-index.json](evidence-index.json).
The copied auth file and runtime descriptor were removed, and no matching
probe process remained after the run. No live host configuration was changed.

The earlier v4 result remains a separate negative run. V5 improves the
evidence boundary by proving partial hook dispatch, but it does not convert
Codex native integration to supported status. A passing acceptance run still
needs complete hook lifecycle capture, durable role coverage, correct seeded
recall, and restart verification under an isolated host configuration.
