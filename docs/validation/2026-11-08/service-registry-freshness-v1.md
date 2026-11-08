# Refresh file-owned authorization before service work

Canonical commit `4e31717a` repairs a reproduced stale-authorization failure in the default persistent memory service. The service previously loaded the scope registry only at startup. Disabling a project, removing the registry or replacing it with malformed JSON did not prevent a later capture carrying the earlier context from being accepted.

Three regression cases failed before the fix because capture succeeded instead of rejecting the stale authorization. Each uses the actual default dispatcher factory with a disposable engine and registry; no live registry is changed. These are service-dispatch checks of an already resolved capture payload, not a native host test or a transport-level replay test.

The default factory now supplies a loader bound to its configured registry path. LocalMemoryService refreshes that registry before dispatch, direct capture and background job processing, and updates the engine's scope-registry reference. A missing registry supplies an empty identity map, so earlier contexts fail validation. Malformed or unreadable registry data fails the operation instead of retaining startup authorization. Explicitly constructed services without a loader retain their caller-owned in-memory registry behavior.

A fourth regression confirms that a queued job cannot begin normal processing after external project disablement, and that a subsequent capture succeeds after the project is re-enabled. This checks authorization at operation/job entry. It does not establish atomic revocation against an external file edit during an already running operation or model call.

Validation: 37 service/registry tests passed in 15.58 seconds, followed by 27 replay, SQLite batching, queued capture, scope and lifecycle tests in 19.41 seconds. Total: 64 distinct targeted tests passed. Black formatting and `git diff --check` passed. Retained private JUnit evidence:

- `.venv/registry-fresh-red-v1.xml`: three reproduced failures before the fix.
- `.venv/registry-fresh-green-v1.xml`: no tests collected because the command named a nonexistent replay test file; not a validation pass.
- `.venv/registry-fresh-green-v2.xml`: 37 passes.
- `.venv/registry-fresh-lifecycle-v1.xml`: 27 passes.

## Acknowledgement-path decision

Do not move authoritative project resolution after the durable receipt under the existing contract. `CodexAdapter._context` resolves the project and checks current profile/workspace enablement before capture. `_event_id` includes the resolved project and workspace IDs. The outbox stores already scoped capture payloads; replay submits those payloads rather than rediscovering the original working directory. Deferring resolution would therefore change pre-capture authorization, event identity and the shape of durable queued work, not merely reorder equivalent operations. The receipt's queued label alone does not authorize capturing content from an unresolved or disabled project.

The fixed authorization reload is a necessary correctness improvement, not an optimization. The identity cache stays inactive; FULL commits and the original 100 ms target remain. The complete-path attribution and older installed package evidence remain bound to source `d9ab39ba`, not this new commit. No package rebuild, installed refresh, publication, live configuration change, performance comparison or pilot day occurred in this increment.
