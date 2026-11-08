# Discriminative support feasibility failed

The pinned DeBERTa candidate selected no facts on the fixed 68-question
development screen. It rejected all 40 answerable questions and correctly
abstained on the 28 absent-answer questions. No support witness reached the
prospective threshold. Sixteen directed contradiction pairs triggered vetoes,
but those vetoes do not explain the answer loss: support selection was already
empty. This candidate is closed and remains outside production and the pilot.

| Measurement | Result |
| --- | ---: |
| Supported facts retained | 0/40 |
| Answerable recall | 0/40 |
| Correct abstention | 28/28 |
| Selected-fact precision | Undefined; zero selections |
| Operational failures | 0 |
| Support-only median | 541.55 ms |
| Support-only p95 | 933.20 ms |

Retention, recall and latency fail. Undefined precision does not satisfy the
precision gate. A trivial all-abstain result cannot qualify through abstention
alone. Support timing includes tokenization, batching, inference and selection;
it excludes retrieval and therefore cannot establish complete-retrieval latency.
No historical percentile was added to it. Loading took 2,394.17 ms and the
separate unrelated warmup took 958.01 ms.

The run follows the [prospective protocol](../../plans/2026-11-07-discriminative-support-feasibility-v1.md):
one pass in lexical key order, 68 questions, 241 measured pairs and one unrelated
warmup pair. CPU float32, four intra-op threads and one inter-op thread were
frozen before execution. The runner was committed at `435c7ebf`; fifteen source,
configuration, input and model-file bindings and all dependency versions were
checked before inference. All fifteen file bindings remain unchanged afterward.
The process exited normally, with all 68 starts and results recorded.

An independent audit recomputed every three-class softmax, support witness,
directed contradiction veto, selected set and grade from raw logits. Every token
count matches preparation. The first audit represented zero-denominator precision
as zero; the final audit correctly records it as null. Both records are retained,
and the failed gate result is unchanged. The first command-line preflight also
remains recorded: package invocation imported an unrelated missing dependency;
the direct entry point repaired it before inference or the execution marker.

The [receipt](discriminative-support-feasibility-v1-receipt.json) binds the verified
private archive containing inputs, separate labels, model metadata, raw outputs,
runner, decision rule, preparation helpers and both audit versions. Large model
assets remain separately retained with pinned revision and hashes. No new model
call was made by the audit, and no semantic retry, threshold search, model
substitution or fresh acceptance case was used.

This result rejects this model/template/decision configuration. It does not
establish that every discriminative method fails. The original comparison and
scores remain unchanged. Critical-fact retention, fresh quality acceptance,
10,000-fact complete latency and actual pilot days remain unproven. No product
default, installed pilot artifact, live configuration, database or publication
state changed.
