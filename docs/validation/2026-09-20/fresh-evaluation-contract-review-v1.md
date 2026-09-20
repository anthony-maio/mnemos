# Fresh evaluation contract review

Source inspected: 13f58a1, with evaluation registry repair committed as
1b3e227. This is a read-only schema/harness audit plus the separately tested
registry repair. No fresh holdout is authored, frozen or executed by this note.

The current 200-session v4 input has been repeatedly observed. Its identifier
and hashes prove which bytes were used, not unseen status. The next evaluation
must retain the original minimums, thresholds and three-repetition requirement.
The running 46-case extraction comparison is development work only.

## Required before a fresh evaluation

1. Introduce a new dataset version. Reuse generic source/body, scope, state-pair
   and v3 lifecycle-target validation. Keep v4's fixed repaired canary session
   IDs and query names confined to v4. Preserve every historical input and
   result. Changing IDs or a version string does not create an unseen corpus.
2. Require source event, query and state-pair identities to be globally unique
   across development and heldout sessions. Fact/session IDs already have
   global checks. Current event uniqueness is session-local; query/pair
   uniqueness covers heldout rows only.
3. Freeze newly authored source events, supported and forbidden meanings,
   lifecycle targets, answerable/no-answer queries and state pairs before
   running the selected production candidate. Retain explicit ambiguity notes.
   Existing fields can express new scenarios; new tag vocabularies are not a
   prerequisite and must not remove required coverage.
4. Audit exact ID and normalized text overlap against prior development and
   observed evaluation data. Bind prior corpus hashes, normalization rules and
   results. Hashes cannot prove semantic independence; template similarity,
   common control text and scenario coverage require a separate review. Keep
   authored synthetic scenarios distinct from actual native conversations.
5. Provide semantic review of actual extracted facts, source support and
   required meaning. The existing exact-text/evidence proxy counts unmatched
   paraphrases as false; it cannot by itself prove the 95 percent semantic
   precision gate. Preserve its historical scores and report any independently
   reviewed semantic results separately with coverage and uncertainty.
6. Run all seven profiles on identical frozen data and budgets. Incomplete
   observations remain incomplete. Do not use new schema validation to promote
   old failures, invent missing correction targets or silently narrow scope.

## Concrete repair completed

The registry builder used setdefault for project/profile and workspace/profile
bindings. Inconsistent authored contexts could silently keep the first profile.
The 1b3e227 repair rejects conflicts, including a foreign profile paired with an
inherited session project, before provider discovery or output creation.
Same-profile workspace sharing remains allowed. Unknown/empty profile selections
also fail before provider probing. Frozen v2-v4 inputs are not rewritten.

The initial focused regression run produced five failures and one pass before
the fix. The expanded test file covers eleven cases after the repair. Final
consolidated verification is recorded separately; this note is not a full
package checkpoint or a model-quality result.

## Relevant implementation

- `benchmarks/local_memory/contracts.py`: dataset versions, parsing, generic
  scope/body checks, v4-specific repaired canaries, identity checks and minimums.
- `benchmarks/local_memory/heldout_run.py`: registry construction, local provider
  discovery, frozen source/model/options bindings and real-model execution.
- `benchmarks/local_memory/ollama_adapter.py`: lifecycle execution and exact-text
  extraction proxy; measurement notes correctly distinguish semantic precision.
- `tests/test_evaluation_registry_identity.py`: conflicting bindings and early
  rejection regressions.

The next source/schema changes need their own tests and review before authoring
or launching a fresh corpus. None of these preparations closes T18.
