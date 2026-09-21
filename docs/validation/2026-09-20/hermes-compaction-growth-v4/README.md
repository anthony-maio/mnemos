# Hermes compaction growth and evidence preservation

The corrected v4 fixture passes 25 checkpoints, including 20 compactions,
reopen, a real user edit, another compaction, and another reopen. Each checkpoint
compares the exact active retained-original origin IDs, roles, content and cwd
against the expected sources, validates every physical evidence component with
the candidate reader, and checks the canonical source rows directly.

Two initial messages become 62 physical message rows after 20 compactions;
there are 60 origin mappings, 62 evidence components, three active rows and
40 rewind rows. SQLite logical allocation increases from 290,816 to 319,488
bytes. After the user edit and further compaction there are 67 messages,
65 origins and 67 components. These are small-fixture storage observations,
not a workload forecast or a comparison against the unmodified upstream base.

Ending the disposable session with end_reason='cli_close' and then using the
normal prune API removes one session and all its messages, origin mappings and
evidence components. The saved post-prune database passes integrity and foreign
key checks. Six candidate files have identical hashes before and after.

Earlier v1/v2 used summary-only compaction inputs and an active-only projection;
those observations cannot establish loss of archived evidence. V3 also failed
to prove exact active-user continuity: its first-user selection could select a
summary on later iterations, and its membership checks accepted archived rows.
V2/v3 pruning did not run successfully because the fixture did not end the
session with the correct API keyword. Preserve these scripts/results as limited
measurements, not passing continuity or pruning evidence. V4 makes the required
assertions explicit and uses actual carried-row metadata with matching tail size.

This tests an isolated Hermes candidate's SessionDB contract. It is not a fresh
interactive host capture, model-extraction result, or a Mnemos-wheel integration
check. Native row generations grow during a long-lived session; no generation
retention policy was invented, and no live data or host configuration changed.
No bounded storage claim follows from successful whole-session pruning.
