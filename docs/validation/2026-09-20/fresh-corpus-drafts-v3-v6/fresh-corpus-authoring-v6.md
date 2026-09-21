# Fresh corpus v6 authoring log

V6 extends the preserved v5 synthetic draft after a semantic question review. V1 through v5 remain byte-for-byte unchanged. This is authored synthetic data, not a real conversation corpus, model output, or human-approved ground truth. The manifest retains `source_label: synthetic_seed`, `human_acceptance: not_run`, and `advertised_as_real: false`.

## Files read

- `benchmarks/local_memory/contracts.py`, for the implemented v5 schema and structural validation rules.
- `.venv/fresh-corpus-authored-v5.jsonl`, the immediately prior authored artifact, for targeted question repair.
- `C:/Users/Ant/.codex/skills/writing-in-anthonys-voice/SKILL.md`, because the workspace writing instruction applies to this log.

I did not read older datasets, prompts, model outputs, handoffs, or source extraction implementation. I did not call Ollama or any evaluated model, use network data, or publish the draft.

## Counts and category design

- 200 held-out sessions and 20 disjoint development sessions.
- 760 total facts: 740 held-out facts and 20 development facts. Held-out status counts are 700 critical supported, 20 superseded noncritical, and 20 forgotten noncritical. Final held-out supported evidence write modes remain 370 explicit and 330 automatic.
- 200 answerable held-out queries and 200 no-answer held-out queries. The no-answer split remains 150 ordinary in-scope unknowns and 50 foreign scope canaries.
- The first 100 held-out sessions each contain exactly six supported same-project facts with separate user evidence. Each answerable query selects one pool member and asks a content-specific question. Reviewer facts are asked as reviewer requirements, command facts as verification artifacts, and database, storage, retry, rollback, and policy facts have their corresponding asserted fields requested.
- 80 state pairs derive one exact content-specific core question from the supported target fact. Both variants retain that same question and append confident versus uncertain affective context; both retain the same target fact and matched content. Optional graph/vector seed lists remain empty.
- Development rows use diagnostic and release-planning forms. Some release vocabulary is shared with held-out rows, so the semantic families are separate in task framing rather than claimed to be disjoint domains.

## Self-audit

The v5 loader passed. Additional checks found zero answerable questions embedding their complete expected answer, zero state questions embedding matched content, all 100 ranking pools at six facts, all 100 ranking queries selecting one pool member, and all 80 state pairs sharing their exact core question and target fact.

The corpus still uses bounded project families and repeated event scaffolding. Content-specific questions remove the earlier missing-value questions and full-answer fallback, but they do not establish semantic novelty or independence from every other evaluation corpus. Identifier changes do not establish unseen independence, and this draft makes no such claim.

## Structural validation

Validated locally with `benchmarks.local_memory.contracts.load_dataset` only. No model inference was run.
