# Fresh corpus authoring log

This draft is an independently authored synthetic seed. It is not a transcription of real conversations, a model output, or a human-approved ground truth set. `human_acceptance` remains `not_run`, and the manifest advertises `synthetic_seed` with `advertised_as_real: false`.

## Files read

- `benchmarks/local_memory/contracts.py`, read for the implemented v5 loader schema and structural lifecycle rules.
- `C:/Users/Ant/.codex/skills/writing-in-anthonys-voice/SKILL.md`, read because authored documentation is required to follow the workspace writing instruction.

I did not read existing datasets, prompts, validation outputs, handoffs, or source extraction code. I did not call Ollama or any evaluated model, publish the corpus, or use a network source.

## Intended and actual counts

- 200 held-out sessions and 20 disjoint development sessions.
- 200 critical supported held-out facts, plus 60 noncritical lifecycle facts (20 superseded and 20 forgotten rows, with replacement facts counted among the 200 critical facts).
- 200 held-out answerable queries and 200 held-out no-answer queries.
- 80 held-out state pairs.
- Hosts are distributed across codex, claude-code, and hermes. Projects cycle through ten developer contexts, each with a unique project ID. Every session includes user, assistant, and system events, ordered ISO timestamps, and a restart event.
- Automatic capture and explicit remember are balanced across the rows. Explicit remember bodies equal the supported fact body. Correction rows replace a prior supported user assertion with a new supported fact; forget rows target noncritical transient facts.
- Scenario tags cover conflicting, hypothetical, assistant_claim, noise, critical_explicit, forgotten_replay, long_session, cross_profile, scope_canary, and correction. Foreign no-answer canaries use a fresh profile and project identity per row so a project ID is never shared by conflicting profiles.
- Family accounting in the manifest records 180 contextual assertion rows, 40 lifecycle change rows, 200 scope canary query cases, and 20 development rows. These are overlapping labels rather than exclusive classes.

## Rubric ambiguities and authoring compromises

The contract requires all sessions to have at least one query and every state pair to resolve to a supported fact in the same scope. To keep those invariants explicit, each row has one local answerable query and one foreign-context no-answer query; the foreign query has no fact IDs and no expected answer. The state-pair family is present in the first 80 held-out rows, including corrected rows retargeted to the replacement fact.

The requested minimum is 100 critical facts, while the session minimum is 200. This draft intentionally supplies one critical project fact per held-out session so the corpus does not meet the minimum through a small repeated fixture. The scenarios still share a compact structural family for reliable parsing; project domains, hosts, wording, lifecycle status, and tags vary, and the manifest records that dependence for later semantic review.

The schema does not provide a separate field for a free-form rationale on each row, so the manifest keeps authoring rationale and the log carries the category accounting. Any claim that the corpus is unseen, semantically independent, or evaluator-clean remains open until the parent performs overlap and semantic review.

## Structural validation

Validated locally with `benchmarks.local_memory.contracts.load_dataset` only. No model inference was run. The passing digest and exact parsed counts are reported to the parent agent.
