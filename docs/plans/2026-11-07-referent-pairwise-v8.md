# Isolated referent-pair comparison v8

Status: prospective. Freeze implementation, data, scoring and protocol before inference.

The retained v7 diagnostic contains 34 distinct literal candidate/target pairs.
Qwen3 assigned differing valid relations across matched views for 15 of them;
Qwen3.5 did so for eight. Twelve and eight pairs respectively were never correct.
Target IDs, order and competing facts varied while each pair's meaning remained
fixed. Removing that variation cannot by itself repair a wrong semantic judgment.

Test an opt-in pairwise path. Reuse the existing referent instructions, value-span
inventory, schema and validation with one target per request. Give that target
the constant local ID `observed-target` and version 1. The runtime retains the
actual target ID, version and scope outside model input and maps each validated
comparison back to its observed snapshot. No original fact is rewritten. The
existing scope, duplicate identity, concurrent-change and operation guards remain.
Any failed request fails the case; no earlier partial correction is returned.
The default comparator and installed package remain unchanged.

Run the frozen `referent-stability-development-v1.json` dataset, SHA-256
`d69661dafbe6c8099dd5a110fcecdce5f2848d1a868f4fe964d42aaf831b21c5`,
once for each of the same Qwen3 and Qwen3.5 packages used in
[v7](2026-11-07-referent-model-comparison-v7.md). This is 72 cases and 198 planned
pair requests per model, 144 cases and 396 pair requests overall. No response is
cached or reused across views. Keep the complete target set visible to the
runtime and its final ambiguity decision. Only the individual model request is
isolated. Do not change semantic labels or use expected labels in requests.

Use the same versioned Ollama 0.40.9 runtime, model manifests/blobs, native
tokenizers, context 8192, output allowance 4096, input cap 3584, reserve 512,
temperature 0, generate route, thinking false, top_k 20, top_p 0.95, min_p 0,
repeat_penalty 1, presence_penalty 0, frequency_penalty 0, and seed 2026110701.
Keep the 90-second provider and 120-second case deadlines. Load and measure Qwen3
first, unload it, then Qwen3.5. Shuffle the full case schedule with the same seed.
Bind model identity, context, slot count and tokenizer before and after each run.
Retain one literal structured smoke check and five tokenizer repeat/roundtrip
checks per model; these are setup evidence and receive no quality credit.

Before inference, test that ID/version aliases, target order and distractors do
not change the request for the same literal pair. Verify all 198 planned requests
and schemas across all 72 views. Keep real IDs and semantic labels out of those
requests. Confirm existing batched prompts and schemas remain unchanged. Count
all prospective pair inputs with each actual native tokenizer before generation.

Retain every raw pair response and request. The relation grader validates the
constant local target ID/version and each literal span, then restores the runtime
ID map before applying the existing 198 relation labels. Missing, malformed,
extra or failed responses receive zero relation credit for the case. Independently
replay original responses through runtime validation, verify every request against
its planned pair, and reconstruct operation, relation, combined and seed-stability
scores. Report model variation despite identical requests separately from semantic
correctness. Stable but wrong judgments remain failures.

Compare descriptively with the retained v7 batched observations under the same
sampling controls. Those are historical controls, not interleaved reruns. Report
native counts, calls, operational failures, wrong accepted targets/additions and
latency, including the extra cost of one request per target. The 72 views are
matched development material, not 72 independent fresh scenarios. Do not pool
them with the original 20-case screen or claim fresh acceptance.

Require all 72 operations and all 198 relations correct, no unsafe acceptance
and no operational failures before considering a separate original-screen and
three-repetition confirmation. This diagnostic does not waive the v7 original
20-case gate, fresh evaluation, extraction or persisted-memory requirements.
Preserve failures, stop owned runtimes, verify ports/processes are closed, and
archive private evidence. No live configuration, pilot day or publication occurs.
