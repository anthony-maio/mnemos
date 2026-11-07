# Live support composition: stable selections, failed latency budget

The QA/contrast intersection completed three actual repetitions of the 44 known
development questions. Every component selection matches the saved diagnostic
in every repetition. The combined support step takes 2,990.80 ms median and
3,984.61 ms p95, so it fails the unchanged 500 ms recall budget before retrieval
is included. This combination remains experimental.

Each repetition produces the following results:

| Treatment | Answerable hits | Clean hits | Correct empty abstentions | Unsupported selected facts |
| --- | --- | --- | --- | --- |
| Extractive QA v2 | 25/27 | 23/27 | 16/17 | 4 |
| Qwen3:14b contrast v5 | 27/27 | 27/27 | 12/17 | 5 |
| Whole-fact intersection | 25/27 | 25/27 | 16/17 | 1 |

The intersection's answerable-hit rate is 92.6% and its abstention rate is 94.1%
on this known set. Repeating these questions does not create 132 independent
quality examples or establish fresh acceptance. The trial-period batch answer
and audit-log reviewer remain omitted. The unmarked retention conflict still
produces the unsupported longer-retention answer. Component agreement has not
resolved these three failures.

## What ran

Source `0f26e914dbd9371c5c2138755dd7f478445e588a` adds an isolated QA worker to
the existing composition and adapters. The controller can enforce its deadline
while CPU inference runs in the worker. Cancellation stops the verified owned
interpreter and makes that transport unusable. This is an experiment worker,
not a new product entrypoint or enabled production retrieval mode.

The run used the pinned DeBERTa QA model on CPU and local Qwen3:14b Q4_K_M under
Ollama 0.40.9. Both selectors received the complete eligible candidate cohort;
QA output never filtered the contrast request. Each contrast question/state
request matched the original saved request before inference. The selectors
retained the original source checks, thresholds, token limits, 15-second contrast
deadline and shared 30-second composition deadline.

All 132 observations completed with assessed status. There were 234 QA source
forwards, one separate unrelated QA warmup, 132 contrast requests and 132 HTTP
attempts. No retry, timeout, operational failure or omitted observation occurred.
The original provider retry policy remained available inside the deadline.

## Timing

| Measured stage | Median | p95 |
| --- | --- | --- |
| Complete support composition | 2,990.80 ms | 3,984.61 ms |
| QA component | 247.85 ms | 346.38 ms |
| Contrast component | 2,738.90 ms | 3,709.75 ms |
| One QA source, including IPC | 124.51 ms | 154.84 ms |
| One QA model forward | 121.86 ms | 149.53 ms |

The composition timer includes snapshot construction, QA transport, inference
and decoding, contrast prompt construction and inference, current-store checks
and the returned intersection. It excludes one-time model loading, fixture
seeding and later grading. No retrieval ran. Component percentiles are measured
separately and should not be added together. The contrast step accounts for
most of the measured time; these results do not show that a faster model or a
different selection policy would preserve quality.

## Verification and retained evidence

The worker preflight reproduced all 78 original eligible tokenizations, rejected
two oversized inputs without inference and passed forced cancellation/cleanup
and rejected-reuse checks. An offline comparison verified identical transport
payloads for the 44 original contrast requests despite intervening optional
provider controls. Black, compilation and 12 documentation checks pass.

The audit reconstructs QA decisions from raw logits, verifies exact rendered
contrast prompts and schemas, checks complete answer distributions and source
citations, and recomputes every intersection and grade. All 132 question/component
groups have stable selections across repetitions. The 44 disposable databases
retain all 82 facts and 82 evidence records unchanged; they contain no source
events or jobs. Their content, versions, scopes, assertion status and evidence
match the frozen projection. Original embeddings, metadata and lifecycle
execution were not reconstructed.

All 105 bound files verify. The runtime rehashed all 16 model/runtime assets
before and after inference. QA exited cleanly, Qwen unloaded, the owned server
stopped and the private port closed. The read-only database audit preserved file
bytes. The private archive has 192 verified members, SHA-256
`3fe14d09f7b142c4ec2e0cce8ac714df90a01b6ff3da06676397cc49d7d3489b`.

The archive also retains the initial setup and auditor failures. The first
freeze rejected changed provider bytes; offline parity verified that the added
options did not change this treatment. The parity helper initially assumed the
wrong total request count. The first auditor duplicated the serialized question
preamble during reconstruction and stopped at the exact prompt comparison.
Correcting those harness assumptions changed no model input, output or score;
inference was run once.

Fresh support evaluation, a solution within the latency budget, current installed
and native validation, and actual pilot use remain open. The inactive pilot has
zero days. This result does not change host settings, production defaults or
publication state.

- [Frozen live protocol](../../plans/2026-11-07-support-intersection-live-v1.md)
- [Aggregate receipt](support-intersection-live-v1-receipt.json)
- [Runtime implementation and saved-output replay](support-intersection-runtime-v1.md)
- [Original saved-selection diagnostic](support-intersection-v1.md)
