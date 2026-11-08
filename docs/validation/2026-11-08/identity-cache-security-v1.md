# Identity cache security invalidation

The inactive project-identity cache prototype missed an ACL-only change to a repository directory. A fresh Windows regression changed only the temporary repository's DACL after the first lookup. The old implementation called authoritative Git once across both lookups instead of twice. The failing result is retained in `.venv/identity-security-red-v1.xml`.

Capture worktree commit `4e3e183d` includes current security metadata in each existing routing-path snapshot. Windows reads the owner, group and DACL as a bounded self-relative descriptor. On other platforms the snapshot adds stat ownership fields. If security metadata cannot be read, the cache discards its entry and delegates to authoritative Git. A change observed across authoritative resolution prevents caching that result. No permission is changed by the reader.

The Windows implementation follows the documented [GetFileSecurityW contract](https://learn.microsoft.com/en-us/windows/win32/api/securitybaseapi/nf-securitybaseapi-getfilesecurityw). Owner/group/DACL reads require the applicable read access; inability to inspect that metadata does not justify reusing the old entry.

All 14 identity-cache tests passed in 11.52 seconds, retained in `.venv/identity-security-green-v1.xml`. Coverage includes the real ACL-only change, unreadable security metadata, an injected change during authoritative resolution, ordinary-file edits, repository/worktree changes, unknown Git selectors and registry revocation. The Windows ACL regression exercises real filesystem metadata; the during-resolution race is simulated. No privileged ownership transfer or POSIX ACL mutation was tested. Targeted Ruff checks passed using the available system executable; Ruff was absent from the retained validation environment. Git diff whitespace checks passed.

This closes one reproduced invalidation gap, not the complete cache contract. The prototype remains isolated and inactive. Its owner still needs to track external Git policy, including configuration files and includes. Git documents conditional includes that depend on repository paths, branches or remote URLs, so a single unchanged generation string is insufficient for normal mutable configuration. See [Git configuration includes](https://git-scm.com/docs/git-config#_includes).

No canonical runtime, installed package, live project setting or memory database changed. No latency benefit was measured. The 100 ms complete capture requirement and all other original goal obligations remain open.
