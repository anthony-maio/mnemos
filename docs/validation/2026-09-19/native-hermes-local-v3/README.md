# Patched native Hermes: scope passes, duplicate capture fails

The original acceptance verdict remains **failed**. This run uses verified
Mnemos `51904d3abdf523e0afb086e956046ca1d5839e92`, the wheel in the adjacent
[package checkpoint](../hermes-safety-checkpoint-v1/README.md), and an isolated
Hermes worktree based on `14a346345486418aa85ed6526549412ee2665ade` with the
recorded CLI cwd compatibility patch. The installed Hermes checkout and live
configuration remain unchanged. Module-origin observations confirm the patched
worktree supplied the host code.

The native CLI turn completed in 24.215 seconds. Explicit provider initialization,
project-scoped prefetch, the expected answer, user/assistant SQLite persistence,
foreign-project exclusion, and effective 65,536-token/non-thinking model options
all passed. Two native raw events persisted; extraction was not part of this
baseline fixture. This is evidence for the patched CLI path, not the unmodified
installation or complete host lifecycle acceptance.

The remaining check rejected two of four bridge capture calls. `sync_turn` and
session-end callbacks reused each event ID, but regenerated `occurred_at`.
Concurrent outbox admission returned `blocked` with `Outbox event identity
already has a different payload`. The other two calls committed. The outbox's
strict collision check is working; the adapter must preserve the original
payload for a repeated occurrence. Weakening that check would hide the defect.

The raw callback responses, immutable helper, model/GPU snapshots, scoped seed
controls, persisted native events, and exact host patch are indexed alongside
this report. Runtime authentication tokens are excluded. A successor must retain
the original checks and record all bridge capture outcomes.
