# Recheck authorization after asynchronous fact staging

Canonical commit `3625c7ec` closes a reproduced authorization race across an embedding await. Checking the registry when a job begins is insufficient if permission changes while provider work runs. Four cases reproduced unauthorized fact creation after file-backed disablement, registry removal, malformed registry replacement and caller-owned in-memory disablement.

The regression runs the normal explicit remember job and real deterministic embedding helper, with a wrapper changing authorization after the embedding await returns. Before the fix, all four cases committed a fact. This is controlled scheduling evidence at the provider boundary; no remote model or live registry is used.

The service now refreshes file-owned authorization and validates the job's context again immediately before `commit_fact_and_job`. The check occurs after proposal staging and outside the per-candidate exception handler, so revocation fails the job rather than silently dropping a candidate and marking the source processed. A failed check leaves staged facts uncommitted, the job incomplete and the captured source unconsolidated.

Validation: all four regression cases failed before the fix. Afterward, 37 registry, service, capture/forget-race and correction-merge tests passed in 11.18 seconds. Black formatting and `git diff --check` passed. Evidence remains at `.venv/registry-await-red-v1.xml` and `.venv/registry-await-green-v1.xml` in the improvements worktree.

This closes the demonstrated await window for the normal staged fact transaction. It does not make external registry edits atomic with a SQLite commit, revoke a provider request already sent, or claim coverage of every other asynchronous operation. The earlier job-entry check remains in place. This is a correctness repair, with no latency or package-validation claim. The last installed wheel and timing run remain bound to `d9ab39ba`; live installations, pilot configuration and frozen evidence are unchanged.
