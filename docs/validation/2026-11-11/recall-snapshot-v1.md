# Consistent recall assembly during concurrent memory changes

Canonical source `73d4a242b27b9ff34da9bda4a1db4545045e5e65` fixes a
reproduced ordinary-recall race. On baseline `ec6d9733`, a second SQLite
connection could forget or correct a selected fact before service response
assembly. Recall still returned the old content, sometimes with empty evidence
or evidence from the correction. Scope, assertion-status and view changes had
the same stale-selection gap. The engine's earlier revision check did not cover
this later service boundary.

The service now rereads selected facts and assembles facts, retained evidence,
source status and literal context within one SQLite read snapshot. Deleted,
revised, moved or newly ineligible selections are dropped. A new correction is
not inserted using the previous revision's rank or context budget. When records
are dropped, the existing host tokenizer or conservative byte fallback repacks
the remaining context. Full/compact responses and memory/captured-evidence
views use the same checks; existing degradation diagnostics remain intact.

The guarantee is a consistent assembly snapshot. A deletion committed after
that snapshot is established can still appear in that response. A subsequent
recall observes the deletion. Other storage backends receive fresh selection
checks but do not acquire SQLite's transactional snapshot guarantee. External
registry-file revocation is not made atomic with a SQLite read.

Twelve initial two-connection regression cases failed on unchanged baseline
source and passed after the repair. An additional failure-path case verifies
that a failed evidence read preserves the caller's uncommitted transaction.
The final targeted suite passed **178 tests**, including all 13 new cases,
existing inspection races, scope/registry checks, retrieval revisions, local
service and MCP responses, compact output, host token budgets, degradation and
related forgetting. Formatting and whitespace checks passed. Focused strict
typing passed with `mypy --follow-imports=silent mnemos/memory_service.py`;
full-project typing was not rerun and prior broader typing limitations are not
declared fixed. A bounded code review found no blocking defect and did not
execute the tests itself.

No model or timing experiment, normal host launch, Jev request, live memory,
pilot refresh or live configuration change was involved. This is a source-only
repair. The private wheel and Codex v13 pass remain bound to `f1314f6d`; they
do not verify this new revision. Exact packaging/installed acceptance and native
acceptance remain separate follow-ups. The original quality, complete timing,
fresh acceptance, daily-use pilot and publication gates remain open.

The public receipt binds both changed files and the private JUnit result at
`.venv/recall-snapshot-tests-v1.xml`. The original source and new tests are
preserved in Git, without rewriting earlier package or experimental evidence.
