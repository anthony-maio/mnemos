# Fresh corpus v4 authoring log

V4 extends the preserved v3 synthetic draft. V1, v2, and v3 remain byte-for-byte unchanged. This remains authored synthetic data, not a real conversation corpus, model output, or human-approved ground truth. The manifest retains `source_label: synthetic_seed`, `human_acceptance: not_run`, and `advertised_as_real: false`.

## Files read

- `benchmarks/local_memory/contracts.py`, for the implemented v5 schema and structural validation rules.
- `.venv/fresh-corpus-authored-v3.jsonl`, the immediately prior authored artifact, for the targeted ranking-pool extension.
- `C:/Users/Ant/.codex/skills/writing-in-anthonys-voice/SKILL.md`, because the workspace writing instruction applies to this authoring log.

I did not read older datasets, prompts, model outputs, handoffs, or source extraction implementation. I did not call Ollama or any evaluated model, use network data, or publish the draft.

## Counts and category design

- 200 held-out sessions and 20 disjoint development sessions.
- 860 total facts: 840 held-out facts and 20 development facts. The held-out set has 800 critical supported facts, 20 superseded noncritical facts, and 20 forgotten noncritical facts. The 800 supported facts include the 300 supported facts from v3 plus five additional supported ranking facts in each of the first 100 held-out sessions.
- 200 answerable held-out queries and 200 no-answer held-out queries. No-answer leakage audit found zero relevant fact IDs and zero expected answers on all 200 no-answer rows. The split remains 150 ordinary in-scope unknowns and 50 foreign scope canaries.
- The first 100 held-out sessions each contain exactly six meaningful, supported, project-scope facts with separate user evidence events. Their answerable query selects exactly one member from that six-fact pool using a question written for the selected requirement. The selected pool facts are related release and operating decisions, rather than unrelated noise.
- 80 state pairs retain the exact same core project-decision question and vary explicit confident versus uncertain affective wording. All optional candidate, direct-vector, and graph seed lists remain empty because no separate association was designed.
- Development rows retain the separate diagnostic and release-planning semantic family introduced in v3. They share the required event structure with held-out rows but are not claimed to be wholly unrelated domains. The manifest records the dependence honestly.
- Across the 800 held-out supported facts, final user evidence write modes are 420 explicit and 380 automatic. The additional ranking facts and lifecycle replacements change the aggregate from the earlier primary-only balance; this is the parsed final count.

## Self-audit

The v5 loader passed. Additional checks confirmed all 100 ranking sessions have pool size six, all 100 ranking queries select exactly one pool fact, all 200 no-answer queries remain empty of answer targets and expected answers, and all 80 pairs preserve the same core question while changing affective state wording.

This draft still uses bounded project families and repeated event scaffolding. That is a recorded limitation for later semantic overlap review. Identifier changes do not establish unseen independence, and this draft makes no such claim.

## Structural validation

Validated locally with `benchmarks.local_memory.contracts.load_dataset` only. No model inference was run.
