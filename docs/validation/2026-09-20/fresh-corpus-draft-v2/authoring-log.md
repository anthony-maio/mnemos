# Fresh corpus v2 authoring log

This is a revised synthetic seed derived from the preserved v1 artifact. It remains authored synthetic data, not a real conversation corpus, model output, or human-approved ground truth. The manifest retains `source_label: synthetic_seed`, `human_acceptance: not_run`, and `advertised_as_real: false`.

## Files read

- `benchmarks/local_memory/contracts.py`, for the implemented v5 loader and lifecycle rules.
- `.venv/fresh-corpus-authored-v1.jsonl`, the v1 artifact authored in this task, so its bytes could be preserved while producing a v2 revision.
- `C:/Users/Ant/.codex/skills/writing-in-anthonys-voice/SKILL.md`, because the workspace writing instruction applies to this log.

I did not read any pre-existing dataset, prompt, validation-output, handoff, or source-extraction files. I did not call Ollama or any evaluated model, use network data, or publish either draft. The v1 artifact was not modified.

## Intended and actual counts

- 200 held-out sessions and 20 disjoint development sessions.
- 240 held-out facts: 200 critical supported facts, 20 superseded noncritical facts, and 20 forgotten noncritical facts. The 20 corrected replacement facts are included in the 200 critical supported facts. The complete dataset has 260 facts after adding the 20 development facts.
- 200 answerable held-out queries and 200 no-answer held-out queries. Of the no-answer queries, 150 are ordinary in-scope unknowns with no target facts or foreign identity, and 50 are explicit foreign-profile/project scope canaries.
- 80 held-out state pairs.
- Automatic capture and explicit remember are balanced at 100 held-out primary assertions each. Explicit remember bodies equal their supported fact bodies.
- 40 correction-tagged rows have a real `correct` event with a prior user target, superseded old fact, and supported replacement fact. 20 forgotten-replay rows have a real `forget` event followed by restart and a post-restart user replay question. The forgotten facts remain noncritical.
- 40 long-session rows contain nine ordered events, including an actual review/follow-up sequence, rather than using a tag on the four-event baseline.
- 40 hypothetical rows contain a hypothetical user event; 40 assistant-claim rows contain an assistant suggestion; 40 conflicting rows contain an assistant suggestion that contradicts the recorded user rule; and the noise tag is retained only where the row has a noise event.
- The 50 scope-canary rows use distinct foreign profile/project identities. The remaining 150 no-answer rows are ordinary in-scope unknowns. Hosts remain codex, claude-code, and hermes across multiple project domains, with user, assistant, and system roles and restart boundaries in every session.

## Self-audit and compromises

The first v2 structural run exposed an ID rewrite omission and an empty scenario tag after removing ungrounded tags. I repaired the fact evidence-event rewrite and retain `noise` as a fallback for the one row whose rotated tags were all removed. The v5 loader then passed. A separate audit confirmed correction tag count equals correct-operation count (20), forgotten-replay tag count equals forget-plus-replay count (20), ordinary versus foreign no-answer counts are 150 versus 50, and long-session rows have nine events.

The content still uses a bounded set of developer project families because the loader contract requires a large repeatable fixture and the draft must remain auditable without model inference. Ten project domains, changing assertion wording, lifecycle cases, host rotation, ordinary unknowns, scope identities, and event sequences provide variation. The manifest records family counts and overlap rather than treating tags as independent categories. Semantic novelty, keyword leakage, and evaluator overlap remain open for the parent overlap and semantic review; this draft does not claim to be unseen merely because its identifiers changed.

## Structural validation

Validated locally with `benchmarks.local_memory.contracts.load_dataset` only. No model inference was run.
