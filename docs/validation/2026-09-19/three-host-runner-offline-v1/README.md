# Three-host runner preparation: offline checks

The new disposable Codex/Claude process helpers and native trace parser pass
46 focused tests, Black, and Python 3.10 strict mypy over 85 production and
benchmark files. These are offline checks. No host or model was launched and
the prospective three-host lifecycle remains NOT RUN.

The process helpers bound each phase to 180 seconds, retain Claude's maximum
$0.50 per-phase budget, reject resources outside the disposable root, remove
temporary authentication on success and failure, and record cleanup failures.
Codex uses the generated `.codex` configuration directory. Its interactive
terminal still requires the controller to exit after the native turn finishes.
The caller must verify package, host, model, protocol, and prior queue bindings.

The historical trace probe reads six retained synthetic v9 sessions: three
Codex sessions and three Claude sessions. It verifies complete source bytes
and records tool names without publishing raw sessions or credentials. All
three Claude sessions provide direct tool-use/result correlation. Codex's
sessions record model-authored `exec` code and MCP completion events with
different IDs, without MCP begin events. The parser preserves those observed
completions with `model_call_correlated=false`; they cannot pass the default
native sequence acceptance check. It does not infer arbitrary JavaScript's
behavior from timing or text matching.

This makes an evidence gap explicit. The three-host runner still needs its
orchestrator and a verified way to correlate Codex's code-mode transport, or
a separately recorded direct-tool invocation mode. The older bounded native
operation and SQLite observations remain evidence at their original scope.
This preparation does not establish complete capture, lifecycle success,
automatic extraction quality, or daily-use acceptance.

`historical-trace-probe.json` binds the exact parser snapshot executed during
that read-only probe. The current source subsequently received formatting and
a regression check that rejects a model call recorded after its alleged tool
completion; `checks.json` binds those final source and test bytes separately.
`evidence-index.json` preserves the raw logs, reports, and executed helper bytes.

Running mypy over the entire working directory encountered duplicate package
copies in untracked historical diagnostic folders. The recorded successful
check instead selects `mnemos` and `benchmarks` explicitly, under the configured
strict Python 3.10 settings. It is not a new full package verification.
