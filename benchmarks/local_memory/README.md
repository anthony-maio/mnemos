# Local memory benchmark preparation

Dataset contract `local-memory-heldout-v5` supports new authored scenarios
without v4's historical repaired-canary ID conventions. It retains the same
minimum coverage, v3 lifecycle targets and generic scope/body/state checks,
and requires globally unique event, query and state-pair IDs across both
development and heldout rows. Existing fact/session uniqueness checks remain.
The base user profile is still `t18-synthetic-profile`; alternate contexts must
use explicit target identities with consistent project/workspace ownership.

Version support is not a new corpus or proof of unseen data. The v5 unit tests
deliberately reuse v4 rows as contract fixtures. A fresh evaluation needs a
separately authored and frozen dataset, an overlap audit against observed data,
semantic review and unchanged acceptance gates. No default dataset is changed
by adding the v5 parser path, and historical v2-v4 inputs/results remain intact.

This directory contains the frozen T18 contract, synthetic seed scenarios, and
the offline runner. The JSONL file is deliberately labeled `synthetic_seed` and
`human_acceptance=not_run`. It is a fixture for protocol and scoring work. It is
not a set of independent real conversations, host acceptance evidence, or a
claim that a local model has completed the evaluation.

The offline runner does not contact Ollama, download models, or construct a
Mnemos engine. `OllamaEvaluatorAdapter` is the separate integration boundary:
it accepts a parent-owned service factory, executes ordered source events, and
closes an isolated SQLite service for each session, repetition, and profile.
The factory must register the dataset project IDs in the resolved scope
registry. Restart creates a fresh service over the same run-specific path.
The adapter reports measured
latency, byte growth, token use, and model and embedding provenance. Missing or
rejected measurements remain visible in the artifact and leave the real-model
gate open. Ollama response counters are recorded when the server returns them;
otherwise token usage stays null. These counters are provider counters, not a
billed cost estimate.

Critical retention is judged from the exact current accepted store after
lifecycle operations. Query-based `critical_retrieval` is reported separately.
Extraction precision is judged per actual runtime fact output. A result is
accepted only when its text exactly matches a supported frozen fact, its stored
evidence resolves to the recorded user event, and its profile and scope target
match. This is an exact-match supported-precision proxy. Unmatched text,
including valid paraphrases, counts as false in this proxy; it is not a semantic
factual-precision judgment. Empty output, assistant claims, and generic substring
matches are also rejected. Model provenance is verified only with exact model names and
64-character digests from the reachable server metadata, plus observed
provider callbacks. Ollama extraction requests pass the typed response JSON
schema through its `format` field. Providers without advertised schema support
keep the prompt-only path and remain subject to the same strict parser.

Dataset version `local-memory-heldout-v2` adds explicit target profile,
project, and workspace fields to source events, queries, and state pairs. The
cross-profile and scope-canary rows exercise those targets; invalid or
unregistered targets stay visible as rejected measurements.

The current lifecycle launcher defaults to `local-memory-heldout-v3`. It
preserves the 800 source events and adds explicit correction/forget target
metadata after evaluator target resolution failed in v2. This is a repaired
rerun of observed labels, not a fresh unseen dataset. The amendment and failed
attempts are recorded in [validation progress](../../docs/validation/2026-09-16/progress.md).

The profile matrix is baseline, full, and one disabled module at a time for
surprisal gating, mutable RAG, affective routing, sleep consolidation, and
spreading activation. Every run requires at least three repetitions. Proportion
metrics use Wilson intervals. Profile differences use paired bootstrap resampled
by session, preserving all repetitions in a sampled session.

State-pair results are paired natural-language query diagnostics in the full
pipeline. The adapter also records a supplemental controlled diagnostic that
embeds the semantic wording once, holds one scoped candidate pool fixed, calls
the production affective classifier for both wordings, and scores with the
production `score_candidate` implementation. Its rank-shift and exact-text
semantic-fidelity fields do not replace the natural-language full-pipeline
metrics. The adapter reads surprisal decisions from completed durable job
results. It also records a frozen counterfactual length/noise shadow policy (at
least 50 characters and not one of a small exact boilerplate set). This shadow
policy is labeled in the artifact and does not claim that the production
service applied it; the applied length-filter decision remains not measured.

The command line prepares an explicit `NOT RUN` artifact when no adapter is
provided:

```text
python -m benchmarks.local_memory --profile full --provider ollama --output run.json
```

Real-model execution, human acceptance of scenario wording, real host workflow,
and production or public-release claims each require separate evidence. An
offline preparation artifact leaves these gates `NOT RUN`; see the validation
records for the status and limits of executed runs.

The bounded developmental profile smoke can be launched explicitly:

```text
python -m benchmarks.local_memory.dev_smoke --output-dir artifacts/dev-smoke
```

It runs `dev-001` once through baseline, full, and each single-module
ablation, writing one versioned JSON artifact per profile. It does not run the
held-out set. The launcher uses deterministic temperature and output limits,
the recorded Ollama model digests, and a temporary isolated SQLite directory.

After parent integration review and code freeze, the resumable held-out
launcher is:

```text
python -m benchmarks.local_memory.heldout_run --model qwen3.5:9b --embedding-model nomic-embed-text:latest --think false --output-dir artifacts/heldout-new --work-dir artifacts/heldout-new-work
```

It verifies live Ollama model digests, binds dataset/config/options/code hashes
into checkpoints, and resumes one profile at a time. The earlier 42-hour
estimate for 200 sessions, three repetitions, and seven profiles came from a
historical developmental smoke rate. It is not a forecast for the current
machine or model. No held-out result is valid until the final artifact and all
partial failures have been reviewed.

These commands describe separate runs from an activated Python environment.
They are not instructions to restart or duplicate the existing frozen queue.
The lifecycle and mechanism launchers default to `http://192.168.86.35:11434`; they
verify model identities from that server. Keep the source, options, output,
and work directories bound to their original run. Changed configurations need
new directories and their own review.

## Graph and affective mechanism checks

[Protocol v2](../../docs/plans/2026-09-17-module-evaluation-protocol-v2.md)
defines the current mechanism evaluation. Its separate `module-mechanism-v1`
fixture contains 50 held-out graph cases with 64 facts each plus scope
controls, and 50 held-out affective pairs. It uses real embeddings and the
production graph builder, requiring targets outside the direct candidate
pool for graph-only recall evidence. Affective scoring holds semantic vectors fixed and measures
classifier fidelity separately. Coverage failures and negative results remain
in the report.

After development checks and a separate source/protocol freeze, the mechanism
launcher is:

```text
python -m benchmarks.local_memory.module_mechanism_run --model qwen3.5:9b --embedding-model nomic-embed-text:latest --think false --split heldout --repetitions 3 --output-dir artifacts/mechanism-new --work-dir artifacts/mechanism-new-work
```

This evaluates mechanisms using directly inserted source-backed facts and
frozen synthetic states. It does not execute the seven-profile lifecycle
matrix or establish extraction quality, natural-language retrieval utility,
native-host acceptance, or production readiness.

## Withdrawn supplement and remaining T18 coverage

The old `local-memory-module-challenge-v1` fixture and `module_challenge_run`
remain available for historical reproduction. The 25-candidate supplemental
queue was withdrawn before execution: intended graph targets were already in
the direct pool, and affective targets lacked independent labels. Its four
profiles cannot replace the seven-profile lifecycle matrix.
`--module-challenge-dataset` validates that legacy fixture's format; it does
not make the fixture suitable for module-utility acceptance. Use protocol v2
for the narrower mechanism questions.

T18 still requires evidence beyond these mechanism checks:

- Complete and review the real-model lifecycle matrix, including failures and
  paired uncertainty. Preserve its singleton-candidate limitations when
  interpreting graph and affective results.
- Execute and review the separate applied encoding comparison described below.
  The existing lifecycle shadow decisions do not measure downstream retention,
  retrieval, storage, or token effects of applying that filter.
- Execute the separate retrieval strategy comparison on identical scoped data
  and budgets, described below. The seven named lifecycle profiles do not
  isolate lexical retrieval from vector retrieval.
- Establish graph and affective retrieval utility separately from mechanism
  correctness, with appropriate competing candidates and independent labels.
  Human scenario acceptance and native-host workflows remain separate gates.

Additional comparators or utility studies need their own frozen protocol and
artifacts. Do not add them to, tune, or reinterpret the existing frozen run to
obtain a desired lift.

## Applied encoding comparison

The separate [encoding comparison protocol](../../docs/plans/2026-09-18-encoding-comparison-protocol-v1.md)
compares the full pipeline, the same pipeline with surprisal disabled, and a
length/noise replacement with the other four modules unchanged. The replacement
applies the existing 50-character rule inside the durable extraction job.
Raw capture remains intact, explicit lifecycle actions bypass the filter, and
completed jobs record the actual policy decision. Missing decisions stay null.

Prepare the three-arm experiment offline from a frozen source checkout:

```text
python -m benchmarks.local_memory.encoding_comparator_run --model qwen3.5:9b --think false --prepare-only --output-dir artifacts/encoding-preparation --work-dir artifacts/encoding-preparation-work
```

Preparation writes `NOT RUN` artifacts with dataset, policy, protocol, and code
bindings. It makes no model calls. Real execution needs separate fresh output
and work directories, live model identity checks, and the existing evaluation
queue to finish. This comparison reuses the v3 labels; it is not a newly unseen
dataset. Passing offline tests or preparing artifacts does not close T18.

The [source and installed-package checkpoint](../../docs/validation/2026-09-18/encoding-comparison-checkpoint-v1/README.md)
records the verified implementation and the three offline preparation artifacts.

## Retrieval strategy comparison

The [retrieval comparison protocol](../../docs/plans/2026-09-18-retrieval-comparison-protocol-v1.md)
separates scoped vector retrieval, SQLite FTS5 BM25 retrieval, the current
hybrid path, and the full production path. The existing baseline combines
vector and lexical candidates before cosine ranking; it is not vector-only.
The new comparison runs full-pipeline ingestion once, then copies the accepted
SQLite state for each query and strategy. Each strategy keeps its own access
writes. Later queries start from the same post-lifecycle state.

```text
python -m benchmarks.local_memory.retrieval_comparison_run --model qwen3.5:9b --think false --prepare-only --output-dir artifacts/retrieval-preparation --work-dir artifacts/retrieval-preparation-work
```

Preparation is offline and produces a `NOT RUN` artifact. Actual execution
requires separate fresh directories and live model identity verification after
the existing frozen evaluation queue permits another run. The v3 labels are
reused; this does not create a new unseen dataset. Query timings exclude
snapshot creation and engine startup and do not establish the T09 warm recall
latency gate. Production retrieval defaults and host configuration are unchanged.

The [retrieval checkpoint](../../docs/validation/2026-09-18/retrieval-comparison-checkpoint-v1/README.md)
records the frozen source checks, installed adapter cases, and offline preparation.

## Original-source replay

The `local-memory-heldout-v6` contract adds an explicit `replay` delivery action.
Its `target_source_event_id` must identify an earlier user capture or remember.
Body, role, source kind, write mode and context must match that source; replay
cannot introduce fact evidence, replacement content or a fact mutation target.
The adapter resends the original operation, payload, runtime event ID and
original source timestamp. The replay row's later timestamp describes delivery
order. Raw operations retain both the delivery action ID and replayed source ID.
Earlier dataset versions reject this operation. V5 identity checks and generic
scope/body checks remain in force. A question about forgotten content is not an
original-event replay. This contract adds measurement capability, not real-model
or native-host evidence.

## Extraction evidence resolution

Extraction evidence resolution was tightened on September 20: every declared
runtime source must resolve to a dataset event before the exact-text support
proxy can accept a fact. Previously a valid source could mask additional
missing or unmapped sources. Raw judgments now include
`source_resolution_complete`. This is a scoring correctness repair, not a
semantic precision measurement. Preserve earlier scores with their original
code revision; do not silently recompute historical reports.

## Performance storage location

Choose the data volume explicitly when measuring capture and 10,000-fact
recall. Durable outbox flushes and SQLite commits depend on that volume.

```text
python -m benchmarks.local_memory.performance --endpoint http://127.0.0.1:11434 --embedding-model nomic-embed-text:latest --cognition-model qwen3.5:9b --output artifacts/performance-report.json --work-dir artifacts/performance-work
```

`--work-dir` must name a new directory. It retains the disposable databases
after success or failure so the run can be inspected. Choose a path on the
volume under evaluation; the relative example uses the checkout's volume.
Without this option, the existing system temporary-directory behavior is
unchanged and the fixture is removed afterward. Reports record the resolved
work path, volume anchor, filesystem device ID, database/outbox paths, and
cleanup policy. The device ID is a filesystem identifier, not a disk-model
or media-type claim. Both existing work directories and existing report
files are rejected. A rejected run cannot replace earlier report evidence.

Report each storage placement separately. An SSD result does not supersede a
hard-drive failure or establish a code speedup. This command calls the model
server for retrieval embeddings, so run it after any serialized model-quality
evaluation finishes. Capture-only diagnostics need no model calls, but remain
separate from the combined retrieval/performance result.
