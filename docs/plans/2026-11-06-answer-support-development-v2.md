# Answer-support development comparison v2

Status: prospective development protocol, frozen before inference. This extends
the [answer-support protocol](2026-11-06-answer-support-protocol.md) after the
first experiment exposed both invalid probability distributions and incorrect
support decisions. Historical receipts and thresholds remain unchanged.

## Question and treatments

Can stricter response formatting, smaller batches or separate answer/source
questions improve this local model's support decisions? Compare four treatments
within one run:

| Treatment | Probability fields | Request grouping | Support question |
| --- | --- | --- | --- |
| legacy-batch | Optional, existing behavior | All retrieved candidates | Joint answer/source selection |
| normalized-batch | Every choice required; sum must equal 1 within 0.01 | All retrieved candidates | Same joint question |
| normalized-candidate | Same required distribution | One candidate per request | Same joint question |
| separate-candidate | Same required distribution | One candidate per request | Answer relation plus one entailment question per source |

"Normalized" names the required response contract. The implementation never
renormalizes values or fills missing probabilities. A selected option must have
the highest probability in the required distribution. Invalid replies fail the
operation and receive no abstention credit. The legacy backend can still turn
an omitted distribution into a one-hot choice through its existing validator;
report how often that happens.

The separate treatment asks its questions in the same request, with the same
model and shared query state. It does not establish independent corroboration.
Acceptance requires both `answers` and an `entailed` source at 0.8 or above.
A confident source contradiction blocks support. The smaller of the two
probabilities is a selection heuristic, not a calibrated joint probability.
No probability threshold changes after inspecting results.

## Data and comparison boundary

Use the unchanged `benchmarks/local_memory/fixtures/answer-support-development-v1.json`,
SHA-256 `2c388cb8a603d4a1463496e0a8243a0a22496026c2a0e1d42dca640ad05d9aef`.
There are 10 synthetic scenarios and 20 queries: 10 answerable, 10 no-answer.
The data has already informed development. It is not a held-out evaluation.

Run one repetition across vector, lexical, hybrid and full retrieval. This
produces 320 observations, representing 20 distinct questions under 16
combinations. Do not treat the combinations as independent samples. Each query
and retrieval arm runs once against the same accepted-fact snapshot; all four
support treatments receive those exact returned IDs in that order. Scope and
embedding settings are shared. Extraction is disabled. Full retrieval explicitly
enables spreading and leaves affective routing off.

Retrieve at most five candidates with the existing 1,200-unit byte context
bound. Randomize retrieval order per query and treatment order per query/arm
using the harness's recorded deterministic seed. The recorded snapshot hashes,
baseline IDs, baseline facts and retrieval latency must match within each group
of four treatment observations. Record every request with a treatment/request
ID, and bind raw provider responses to that ID, including invalid responses.

## Runtime and limits

- Private loopback Ollama server on `127.0.0.1:11436`, version recorded before
  running, using the existing private model cache. No live host configuration
  or memory database participates.
- `qwen3:4b`, Q4_K_M, expected digest
  `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`.
- RTX 4080 with 16,376 MiB reported VRAM. Record server environment and loaded
  model context in a preflight receipt. Set context to 16,384, parallel requests
  to one, maximum loaded models to one. Warm-load without a question before
  measuring; record warm/cold limitations.
- Temperature 0, thinking disabled, at most 2,048 output tokens. No claim of
  bitwise deterministic inference. `BAAI/bge-small-en-v1.5` embeddings use the
  existing private cache and the same provider instance across treatments.
- The support operation has a shared 15-second deadline and 64,000-byte total
  serialized request limit, including candidate-at-a-time calls. A later call
  failure makes the entire observation unavailable; earlier traces remain.
- Each final rendered prompt is capped at 12,288 UTF-8 bytes before inference.
  This conservatively reserves 2,048 output and 2,048 template/special-token
  positions within the configured 16,384 context for this byte-level model.
  It is not a measured tokenizer count. Verify actual provider input/output
  token counts and server logs for truncation after the run.

Use a new output directory. Freeze source/protocol in a commit before inference;
the manifest records that commit and source hashes. Stop on runtime identity
mismatch before observations. Archive the protocol, dataset, raw requests,
responses, observations, summary, manifests and accepted-fact databases. Never
overwrite the earlier runs. Unload and stop only the private server started for
this comparison when finished.

## Reporting and next decision

Report each treatment and retrieval arm separately: answerable recall,
unsupported returned facts, correct abstention, false answers, invalid replies,
timeouts, partial results, request count and support latency. Separate returned
fact correctness from an answerable hit that includes extra unsupported facts.
Distinguish actual provider token counts from the trace's character estimates.
Verify every attempted request and observation is accounted for; retain failures.

Compare treatments within this run. The explicit context setting and newer
source mean the earlier experiment is diagnostic background, not a controlled
before/after estimate. No tuning, repeated seeds or preferred-arm-only reporting
inside this run. Any follow-up gets a new protocol and output namespace.

This development result cannot promote a default. The original 90% answerable
recall and 90% abstention targets, fresh 200-scenario evaluation with three
repetitions, separate extraction evaluation, native lifecycle acceptance and
real daily-use pilot remain required.
