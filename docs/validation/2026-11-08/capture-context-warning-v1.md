# Raw capture context warning

The delivery renderer could omit its raw-conversation warning when a `captured_passthrough` record carried an asserted or conflict assertion label. The asserted case rendered like an ordinary accepted fact. Answer-support snapshot eligibility already rejects raw captures independently of that label, and retrieval views distinguish raw captures from accepted facts; context rendering was inconsistent with those boundaries.

The renderer now preserves `captured conversation; unverified, may be outdated` for every raw capture. A conflict label also retains `unresolved conflict`. The change does not mutate stored metadata, promote records, filter additional candidates or alter source-view membership. Complete rendered warnings remain part of the measured context budget. Ordinary asserted facts keep their existing rendering.

The new parametrized regression reproduced two failures before the fix; the existing unknown-label behavior passed. After the fix, all 65 selected context-budget, retrieval-view, adapter-authority, CLI-authority, support-context and compact-retrieval tests passed in 12.55 seconds. Black and diff checks passed. Retained JUnit files are `.venv/capture-warning-red-v1.xml` and `.venv/capture-warning-green-v1.xml`.

This is a local rendering correction for a reproduced inconsistent-metadata case. It does not establish that any live stored record has these labels, verify semantic answer support, meet the latency gates or complete the original pilot. Live databases and host configuration were not changed; no package was rebuilt or published.
