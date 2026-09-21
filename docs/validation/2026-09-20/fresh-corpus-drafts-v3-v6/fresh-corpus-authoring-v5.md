# Fresh corpus v5 authoring log

V5 extends the preserved v4 synthetic draft. V1 through v4 remain byte-for-byte unchanged. This is authored synthetic data, not a real conversation corpus, model output, or human-approved ground truth. The manifest retains `source_label: synthetic_seed`, `human_acceptance: not_run`, and `advertised_as_real: false`.

## Files read

- `benchmarks/local_memory/contracts.py`, for the implemented v5 schema and structural validation rules.
- `.venv/fresh-corpus-authored-v4.jsonl`, the immediately prior authored artifact, for the ranking-pool repair.
- `C:/Users/Ant/.codex/skills/writing-in-anthonys-voice/SKILL.md`, because the workspace writing instruction applies to this log.

I did not read older datasets, prompts, model outputs, handoffs, or source extraction implementation. I did not call Ollama or any evaluated model, use network data, or publish the draft.

## Counts and category design

- 200 held-out sessions and 20 disjoint development sessions.
- 760 total facts: 740 held-out facts and 20 development facts. Held-out status counts are 700 critical supported, 20 superseded noncritical, and 20 forgotten noncritical. Final held-out supported evidence write modes are 370 explicit and 330 automatic.
- 200 answerable held-out queries and 200 no-answer held-out queries. The no-answer split remains 150 ordinary in-scope unknowns and 50 foreign scope canaries.
- The first 100 held-out sessions each contain exactly six supported, same-project facts with separate user evidence. The v3 generic competitor was removed so the pool is exactly six rather than seven. Each answerable query selects exactly one pool member and asks a question derived from that member's actual content: database choice, filter storage, retry policy, rollback note, reviewer, verification command, and similar recorded decisions.
- 80 state pairs derive one exact core question from the supported target fact. Both variants retain that same question and append confident versus uncertain affective context; both retain the same target fact and matched content. Optional graph/vector seed lists remain empty.
- Development rows use diagnostic and release-planning forms. Some release vocabulary is shared with held-out rows, so this log records a separate task framing rather than claiming disjoint semantic domains.

## Self-audit

The v5 loader passed. Independent local checks found pool sizes exactly six for all 100 ranking sessions, one selected target for each of those 100 answerable queries, exact shared state-pair cores for all 80 pairs, and zero no-answer queries whose text exactly contains a supported fact body. A token-overlap inspection found no no-answer query sharing more than one nontrivial token with any supported fact body; the ordinary unknowns and foreign canaries therefore do not repeat a supported answer phrase.

The corpus still uses bounded project families and repeated event scaffolding. Content-specific questions reduce generic query ambiguity, but they do not establish semantic novelty or independence from every other evaluation corpus. Identifier changes do not establish unseen independence, and this draft makes no such claim.

## Structural validation

Validated locally with `benchmarks.local_memory.contracts.load_dataset` only. No model inference was run.
