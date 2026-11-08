# Project identity cache prototype

The capture candidate now has an explicitly owned, one-entry identity cache with routing invalidation tests. It is not connected to the adapter, installed in a pilot, or promoted to delivery. There is no new latency claim.

The snapshot checks ancestor repository markers, file contents rather than only timestamps, directory identities, linked-worktree Git directory pointers, common-directory pointers, local configuration and HEAD. Environment and Git executable identity changes also invalidate the entry. Unsupported routing files and reparse points fall back to authoritative resolution. A failed refresh clears the old entry, and a routing change during resolution prevents caching the result.

Identity is not authorization. The cache stores no enablement, aliases, profile membership or workspace permissions. A test primes the identity cache, disables the project, and verifies that registry validation still rejects it.

External Git policy is an explicit unresolved requirement. The owner must supply a generation covering system/global configuration and includes, executable policy and other relevant Git configuration. Without a generation the cache always invokes authoritative resolution. A constant generation would be unsafe in a changing environment. No provider is implemented or wired into the worker yet; this prototype must not be enabled until that requirement has coverage. The routing snapshot alone does not establish full Git policy equivalence.

Validation: 35 tests passed across the new cache tests, existing project identity resolution tests and scope-context tests. Strict type checking passed for the new module. Coverage includes stable reuse, nested repositories, linked-worktree pointer changes with preserved timestamps, broken common-directory routing, missing Git, environment/policy changes, changes during lookup, and revoked registry enablement.

The first run had four fixture failures: the chosen temporary directory was inside a Git worktree, invalidating three existing plain-directory assumptions, and truncating Git's hidden Windows pointer file failed. The corrected run used a fresh system temporary directory and an existing-file update for that pointer. The first run's files remain intact. No product behavior was changed to accommodate those failures.

Next: establish a complete, inexpensive external-policy generation mechanism, or reject cache integration if its invalidation contract cannot be satisfied. Only then measure full command latency on a newly built package. The original timing and quality failures remain unchanged.
