# Canonical integration of recovered sleep settings

Canonical commit `7c6dd114e85ff04aa94f5673fc34b407ba7373ba` ports the relevant
configuration fix from the recovered local 0.7.0 checkpoint `cc5ecf68`.
It is pushed on `wt/backup-20261108-delivery`; the original dirty checkout is
unchanged. PR #6 and patch PR #9 remain separate review/delivery surfaces.

SleepSettings inherits canonical SleepConfig, preserving its extraction_mode
field and constraints. Every field now accepts a nonempty MNEMOS_SLEEP_* env
override and reaches MnemosConfig.sleep. TOML persistence encodes strings with
JSON escaping so the tested multiline prompts, quotes, tabs and backslashes
round-trip. Global/project/environment precedence remains intact. No profile,
authorization, write-firewall or module-default implementation changed.

The recovered 24 configuration tests were adapted to include canonical
source_bound extraction. Before the fix, 22 failed and two passed. After the
fix, 63 selected settings/runtime/judgment tests passed. Two further tests cover
the new extraction-mode environment path: sleep settings do not enable
experimental modules, and judged extraction still requires a provider. Final
selected suite: 65 passed in 9.13 seconds. A separate 29 UI, native-pilot-settings
and Ollama-generation-settings checks passed. Black accepts the changed Python
files, mypy accepts both runtime modules and git diff --check passes.

Read-only review found no material issue in configuration propagation,
precedence, provider guards or preserved safety/profile behavior. The reviewer
did not run tests or inspect live settings. Documentation explains these as
experimental SleepDaemon controls, not a measured quality benefit.

Private red/intermediate/final logs and JUnit are retained under
`.venv/canonical-sleep-*-v1.*` in the improvements worktree. These 94 final checks
are source validation, not a full-suite result, installed artifact, native run,
pilot use or publication. Existing wheels retain their old hashes and do not
contain this fix. No live configuration or memory database was changed.

The recovered episode identity/buffer deduplication repair remains a separate
next integration. It must preserve canonical profile partitioning and the write
firewall, and must not claim content-based hook deduplication proves occurrence
identity. Semantic quality and complete latency obligations remain unchanged.
