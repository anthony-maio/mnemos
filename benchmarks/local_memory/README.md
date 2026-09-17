# Local memory benchmark preparation

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
match. This is an exact-match supported-precision proxy. Paraphrases are
unjudged by this proxy; empty output, assistant claims, and generic substring
matches are rejected. Model provenance is verified only with exact model names and
64-character digests from the reachable server metadata, plus observed
provider callbacks. Ollama extraction requests pass the typed response JSON
schema through its `format` field. Providers without advertised schema support
keep the prompt-only path and remain subject to the same strict parser.

Dataset version `local-memory-heldout-v2` adds explicit target profile,
project, and workspace fields to source events, queries, and state pairs. The
cross-profile and scope-canary rows exercise those targets; invalid or
unregistered targets stay visible as rejected measurements.

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
and production or public-release claims remain `NOT RUN` until their separate
evidence is observed and recorded.

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
.venv\\Scripts\\python.exe -m benchmarks.local_memory.heldout_run --output-dir artifacts/heldout --work-dir artifacts/heldout-work
```

It verifies live Ollama model digests, binds dataset/config/options/code hashes
into checkpoints, and resumes one profile at a time. A full 200-session,
three-repetition, seven-profile run is expected to take roughly 42 hours at
the observed serial developmental smoke rate; parallelism or faster hardware
changes this estimate. No held-out result is valid until the final artifact
and all partial failures have been reviewed.

Pass `--module-challenge-dataset` to validate the supplemental fixture before a
module-utility run. That fixture has 50 held-out state pairs and 25 scoped
candidates per pair, including declared graph targets. Its controlled module
challenge remains a separate `NOT RUN` gate until those rows are executed and
their production traversal traces are recorded. The lifecycle v2 run remains
independent and keeps its singleton-row limitation visible in interpretation.

The executable challenge command is:

```text
.venv\\Scripts\\python.exe -m benchmarks.local_memory.module_challenge_run --output-dir artifacts/module-challenge --work-dir artifacts/module-challenge-work
```

It runs baseline, full, affective ablation, and spreading ablation for three
repetitions, with resumable per-profile checkpoints. It is expected to take
about 35 to 70 hours serially, extrapolated from observed developmental Ollama
response and embedding timings. Hardware and model load state can move this
estimate. The artifact
keeps actual direct seed IDs, candidate ranks, graph target observations, and
the explicit status when a target was already in the direct pool.
