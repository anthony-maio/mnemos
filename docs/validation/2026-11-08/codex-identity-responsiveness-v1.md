# Codex identity discovery and worker responsiveness

The Codex adapter now runs project identity discovery through `asyncio.to_thread`. It still performs authoritative Git discovery for every lookup and retains the existing timeout, retries, environment filtering and stdin isolation. No identity cache is activated.

Previously, the synchronous lookup ran inside the resident hook worker's event loop. A stalled Git call could prevent unrelated async work and cancellation from progressing until discovery returned. Two regression cases reproduced that behavior with a bounded, gated lookup: both failed on the prior source because the hook task finished before the waiting observer could run.

The new cases verify that the event loop progresses while discovery remains pending, that no client retrieval starts before resolution, and that cancelling the hook discards a late result without capture or retrieval. Normal completion still makes the expected retrieval call. Payload identity fields are copied before dispatch to the thread.

All 59 tests passed in 35.29 seconds across the new cases, Codex adapter, project identity resolution, resident hook worker and SQLite hook worker suites. Black and diff checks passed. Original failing JUnit is `.venv/identity-responsive-red-v1.xml`; passing JUnit is `.venv/identity-responsive-green-v1.xml`. No failing test was removed or relaxed.

This is event-loop responsiveness evidence, not a reduction in Git subprocess time or proof of the 100 ms complete capture requirement. Cancelling an awaiting task does not kill the discovery thread or Git child; the existing bounded discovery timeout still governs their completion. Resident hook dispatch remains serialized by its existing lock. Claude's separate adapter path is unchanged and requires its own follow-up. The change is local source only; frozen packages, live hosts and the optional explicit-memory pilot are unchanged.
