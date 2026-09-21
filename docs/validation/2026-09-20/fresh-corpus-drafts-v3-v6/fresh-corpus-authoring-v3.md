# Fresh corpus v3 authoring log

V3 is a new synthetic draft built from the preserved v2 artifact. V1 and v2 remain byte-for-byte unchanged. This remains authored synthetic data, not a real conversation corpus, model output, or human-approved ground truth. The manifest retains `source_label: synthetic_seed`, `human_acceptance: not_run`, and `advertised_as_real: false`.

## Files read

- `benchmarks/local_memory/contracts.py`, for the implemented v5 schema and lifecycle validation rules.
- `.venv/fresh-corpus-authored-v2.jsonl`, the immediately prior authored artifact, for targeted repair while preserving v1 and v2.
- `C:/Users/Ant/.codex/skills/writing-in-anthonys-voice/SKILL.md`, because the workspace writing instruction applies to this authoring log.

I did not read older datasets, prompts, model outputs, handoffs, or source extraction implementation. I did not call Ollama or any evaluated model, use network data, or publish the draft.

## Counts and category design

- 200 held-out sessions and 20 disjoint development sessions.
- 320 total facts: 300 held-out critical supported facts and 20 development supported facts. The held-out set has 20 superseded noncritical facts and 20 forgotten noncritical facts in addition to its 300 supported critical facts.
- 200 answerable held-out queries and 200 no-answer held-out queries. The no-answer split remains 150 ordinary in-scope unknowns and 50 foreign scope canaries.
- 80 state pairs. Each pair asks the same underlying project decision question with explicit confident versus uncertain affective wording. Candidate, direct-vector, and graph seed lists are empty because no separate association was designed.
- The first 100 held-out answerable queries each rank two genuinely supported, same-project facts: the primary decision and a competing related requirement. Each competing fact has its own user evidence event. The remaining answerable queries target one supported fact.
- All 20 correction rows now target the actual earlier `-capture` user assertion. The generic placeholder target event was removed. Each replacement changes a substantive value, and the correction event supplies the replacement evidence and readable old/current assertion.
- Automatic and explicit primary capture remain balanced at 100 each. Added ranking facts alternate capture and explicit remember modes, keeping the complete held-out supported-fact mix balanced at 150 automatic and 150 explicit assertions.
- Development rows use a diagnostic and release-planning semantic family with distinct recorded content and query wording. They retain the same structural contract, but are not claimed to be 20 unrelated domains. Held-out rows retain the varied project-operation family from v2.

## Self-audit

The v5 loader passed. Additional local checks confirmed 20 of 20 correction tags target an actual capture event, all 20 replacement values differ materially from their superseded values, all 100 ranking queries include two supported facts, all 80 state pairs contain both explicit affective states and no optional seed IDs, and development content uses its separate family.

The dataset still has a bounded set of project families and repeated event scaffolding. That dependence is intentional and recorded in the manifest; it is a limitation for later semantic overlap review. Identifier changes do not establish unseen independence, and this draft makes no such claim.

## Structural validation

Validated locally with `benchmarks.local_memory.contracts.load_dataset` only. No model inference was run.
