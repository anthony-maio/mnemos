# Hermes compression and edit provenance repairs

Four new route/role cases failed because unmarked compression output acquired
retained-original authority. The repair classifies unproven compression output
as derived context while retaining actual original rows and ordinary writes.
Tests cover in-place and child compression, user and assistant roles, restart,
and repeated compression with stable canonical identities.

An actual live-payload edit inside a hybrid wrapper reproduced stale ancestor
provenance: the reader rejected the changed component against the unchanged
parent. The repair writes the revision as a separate inactive, non-compacted
source row within the same transaction. The original parent remains unchanged;
the revised component points to matching text and survives restart and recopy.
The source record does not add an active or displayed conversation turn.

A further failing test showed editing an unmarked derived user carrier could
promote it to retained-original. Refresh now preserves derived classification,
and full-row edits bind to the edited row rather than an unchanged ancestor.
Existing evidence is validated before edits, and unchanged writes preserve
identity. Injected failure after source insertion rolls back transcript,
components, and the new source row; tampered source evidence remains rejected.

The combined focused selection passes 32 tests. The full SessionDB rerun was
still in progress when this focused archive was collected. General host
acceptance, native delivery, complete capture, and full-suite acceptance remain
open. This code is only in the isolated host candidate, not live Hermes.

Failed attempts are preserved. The initial v22 test incorrectly demanded empty
evidence instead of derived context; the hybrid draft incorrectly expected old
content after editing. Corrected tests assert the actual provenance contract,
and the original drafts remain outside executable test directories. The v23
repair also exposed a keyword incompatibility in a rollback test's injected
handler; ordinary writes now retain the previous calling convention.
