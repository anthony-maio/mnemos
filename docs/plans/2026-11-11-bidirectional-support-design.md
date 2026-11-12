# Local bidirectional support decision

The native artifact gap is closed for the explicit Codex preset. The next
semantic work must address local support quality and the complete clock, not
repeat packaging, output formats or threshold searches. Required local operation
and optional explicit Jev remain unchanged. Every closed candidate stays closed.

## Decision and alternatives

Investigate one small, fully bidirectional encoder with a shared per-target
head and later task-specific supervision. Use the Apache-2.0
`tasksource/ModernBERT-base-nli` backbone, with its published NLI classification
head discarded. The publisher describes it as a further-fine-tuning backbone.
Its inherited NLI scores must never be substituted for answer-support scores.
The base architecture has 22 layers, 149 million parameters, alternating local
and global attention, and native 8,192-token context. These are publisher
properties, not measured Mnemos support quality or speed.

This changes the failed causal-prefix/rank-8-adapter architecture: all positions
are bidirectional, the backbone is substantially smaller, no vocabulary head or
generation is used, and future task training would update the full encoder and
support head. One pass covers the query, date, complete retrieved cohort and
every candidate's owned retained sources. A shared head reads trusted per-target
marker positions. No candidate is judged from an isolated generic NLI template.

Reject another generative output-format experiment: it does not address the
recorded semantic or computation failures. Defer a symbolic source compiler:
it still needs reliable unrestricted query/source parsing, and must not treat
quoted circumstances as adopted constraints. Neither alternative narrows the
original request contract or changes the existing gates.

## What this decision authorizes

One capacity-only screen, before any training or semantic inference. It reads
synthetic workloads without labels and uses a seeded random shared head solely
to measure computation. Pin the model repository revision and every downloaded
file hash before loading, use safetensors only and trust_remote_code=false.
No user/project facts, Jev request, remote inference, deployment, live
configuration or pilot refresh is included. Model acquisition is private and
bounded to one repository, tokenizer/configuration and one weight file.

Freeze CUDA 0 (RTX 4080), float16, SDPA, reference_compile=false, inference/eval
mode, one request, four intra-op threads, one inter-op thread, seed 2026111101,
TF32 disabled and no hidden-state history/attention outputs. Reuse the retained
torch 2.10.0+cu128 / transformers 4.57.6 environment without dependency changes.
Require a complete pretrained backbone load; missing backbone weights fail.
Record actual GPU, driver, package versions and pre-existing memory use.

Preserve the existing strict cohort serializer's query, date, opaque IDs,
versions, assertion status, source ownership and exact source text. Add the
model's trusted CLS/SEP tokens outside serialized content and shift trusted
marker positions accordingly. Admission is zero through five candidates,
64,000 payload bytes and 8,192 total tokens including all boundary/decision
tokens. Oversize/invalid inputs fail the entire request, without truncation or
omitting a source. This experimental bound is not a reduced product promise;
future evaluation counts unavailability against its original denominators.

Use six prospectively fixed synthetic workloads: empty, one short candidate,
five short candidates, five candidates at the exact token bound with one source
each, the token bound with eight sources each, and the exact byte bound. Include
token/byte overflow rejection before model execution. After one unrelated
warmup and one warmup per nonempty workload, measure twenty round-robin samples
per workload. Each measurement includes validation, serialization, tokenization,
device transfer, forward computation and host readback. Record every sample
and both finite shape-correct outputs and failures; scores are not decisions.
End each timed operation only after CUDA synchronization and materialized,
shape-checked, finite CPU outputs. Any operational, nonfinite or binding failure
fails the whole attempt regardless of its measured p95; missing samples never
improve the admission grade.
Verify 106 actual root-model forward calls: six warmups and 100 measured
nonempty requests; the twenty empty requests perform framing only. Use the
19th sorted duration of twenty for p95 (nearest rank), without trimming samples.
Twenty measurements are a small computational screen, not a tail guarantee.

The necessary admission target is p95 <= 200 ms on every measured workload,
leaving 300 ms for the rest of the unchanged 500 ms complete-recall requirement.
That remainder is a budget, not an established measurement. Cold loading and
first-use are separate. A failed capacity result closes this exact configuration
without a shorter bound, different depth/dtype/attention backend or timing retry.
An outer owned worker bound is 600 seconds; inference is a single attempt with
an exclusive marker. Rehash bindings and stop owned processes afterward.

## Requirements after a capacity pass

A pass does not authorize training by itself. First prepare and review a
specific learning/data plan with new grouped development data, provenance,
training-only labels, a fixed objective/optimizer/selection rule, and explicit
subject/relation, negation, attribution, requested/unknown conditions, temporal,
conflict and evidence-ownership coverage. Preserve the old 127-case corpus and
every score as known diagnostics. Do not call it fresh acceptance or assume its
three connected training groups provide enough data. Model-card benchmarks
and inherited NLI training are not evidence of Mnemos support quality.

Before production integration, preserve the existing scope/authorization and
fact/evidence revision checks around inference. A future resident worker must
not reload weights per recall or block the owner event loop. Its queue wait,
source snapshot, tokenization, decisions and revalidation belong inside complete
timing. Keep Jev explicitly selected and separate, never a local-failure fallback.
The original 95% precision/critical-retention, 90% recall/abstention, fresh
acceptance, complete timing, native/pilot and publication requirements remain.

Primary sources checked November 11:
[publisher checkpoint](https://huggingface.co/tasksource/ModernBERT-base-nli),
[base architecture](https://huggingface.co/answerdotai/ModernBERT-base), and
[Transformers implementation](https://huggingface.co/docs/transformers/model_doc/modernbert).
