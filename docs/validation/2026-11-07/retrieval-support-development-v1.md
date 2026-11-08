# Actual retrieval plus support fails the development gates

The fixed QA/contrast treatment substantially reduces unrelated selections,
but does not qualify for promotion. On 68 known questions repeated twice, it
retains 72 of 80 expected answers, selects four unsupported facts and correctly
abstains on 52 of 56 absent-answer observations. Precision is 94.74%, below the
95% target. Expected-fact retention is 90%, also below 95%. Complete p95 is
8,641.48 ms against the 500 ms development screen.

This is one completed run under the
[bounded protocol](../../plans/2026-11-07-retrieval-support-development-v1.md).
There was no integration rerun, prompt adjustment, threshold change or new
acceptance data. The evidence audit passed and all owned processes stopped.

| Measurement | Baseline retrieval | Retrieval plus intersection |
| --- | ---: | ---: |
| Answerable hits | 80/80 | 72/80 |
| Clean answerable hits | 14/80 | 72/80 |
| Expected facts retained | 100% | 90% |
| Selected-fact precision | 31.50% | 94.74% |
| Correct abstention | 0/56 | 52/56 |
| Unsupported selections | 174 | 4 |
| Operational failures | 0 | 0 |
| Complete median | 136.53 ms | 6,905.58 ms |
| Complete p95 | 293.36 ms | 8,641.48 ms |

All absent-answer observations had available candidates. Every expected answer
was present in retrieval's top five; the eight lost answers were rejected by
support selection. The treatment reaches the 90% Recall@5 and abstention point
targets, but still fails precision, retention and latency. Neither arm meets
all development gates.

The 95% scenario-cluster bootstrap intervals are 86.49-100% for treatment
precision, 80.00-97.62% for retention/Recall@5, and 81.48-100% for abstention.
The median paired latency increase is 6,757.61 ms, with interval
6,640.41-7,020.88 ms. These use 10,000 resamples of the 32 scenarios and the
prospective seed. Two repetitions do not create 136 independent questions.

## Artifact and comparison boundaries

The runner at `49044983` imports the installed `f5a944ea` wheel before adding
benchmark code to its import path. All 96 installed package files and 57
dependency versions match the frozen installation. The protocol uses actual
`LocalMemoryService.retrieve`, baseline profile, memory view, project/global
scope, top-k five and a 2,048-unit UTF-8 byte budget.

The 32 original scenarios contain 61 declared facts, including scope controls
and stored conflicts. Every observation starts from its own copy of the full
scenario snapshot. Both arms return identical candidate IDs and baseline
context in all 136 pairs. The treatment receives only those IDs. The audit
checks all 272 database copies, retained sources, final context, QA spans,
contrast responses and intersection decisions. No extraction runs.

There are 246 measured QA source evaluations, one separately recorded QA
warmup, 136 contrast requests and 136 HTTP attempts. Model loading and embedding
warmup are outside the measured spans. The fixed model/configuration bindings
and all 33 model assets remain unchanged. The audit makes no quality-model
calls. The first preparer failed on SQLite tuple-row handling before any
measured observation; its source, log and partial snapshot are retained.

## Context cost and limits

Returned context totals 25,530 byte-bound units for baseline and 7,410 for the
treatment. Supported facts per 1,000 units rise from 3.13 to 9.72 while expected
fact retention falls. These are conservative UTF-8 byte bounds, not measured
host tokens. The treatment additionally consumes 101,228 Qwen prompt tokens,
17,666 generated tokens and 10,158 QA source-input tokens. Different tokenizer
units stay separate. No net token-saving claim follows from shorter context.

These small known scenarios do not establish 10,000-fact latency or fresh
quality acceptance. Their original labels have no separate critical-fact
annotation, so critical-fact retention remains unmeasured. This run adds no
native-host or daily-use evidence. The proposed baseline pilot remains inactive
at zero days; treatment results cannot qualify that different configuration.

The [receipt](retrieval-support-development-v1-receipt.json) binds 499 verified
private archive members, SHA-256
`1911c50ecce4e8e5911ed5142208dd9a9810da5db7cdb81a7f1fabe2a904d86e`.
Runtime checks preserve all 127 frozen input/source files and the watched
original paths. QA exits normally; the owned Ollama tree is stopped after model
unload, and its port closes. Live configuration is unchanged, and nothing was
published.

The updated release checklist and plans pass all 12 documentation checks with
the explicit canonical website checkout. The first check invocation skipped
the website test because that path was unset; its original log is preserved.

## Decision

Keep this treatment outside default retrieval and the pilot. The run does not
show an integration defect that would justify using the permitted rerun.
Close this bounded comparison with its failed result. Do not spend fresh cases
on it or reopen output-codec tuning.

The next independent work is the installed-pilot timing protocol and normal
native hook validation. An explicit MCP-only learning pilot can be prepared as
a separate proposal, with automatic recall/capture disabled and evidence
inspection required. It needs a prospective user decision and cannot satisfy
the original full-pilot or semantic-quality requirements. A further semantic
candidate needs a separate design decision that addresses both answer loss
and complete latency; changing another output encoding does not do that.
