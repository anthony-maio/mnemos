# Packed query probe v1

This is a query-only development probe with supplied source annotations. It cannot establish extraction quality, complete retrieval latency, fresh acceptance or pilot readiness. The earlier failed screens remain closed.

The fixed 16-case schedule covers owner/region association in both candidate orders, unrelated fields and subjects, conflicting values in both orders, an applicable/excluded/unknown condition, current and historical questions against undated evidence, two supplied support controls, and an instruction embedded in source text. Source content is synthetic. Cases 14 and 15 deliberately override the source-support disposition to test control handling; exclude them from semantic accuracy and report them separately. All other cases also use authored annotations, so successful query interpretation would not prove that the extraction pipeline can produce those annotations.

Before inference, save the complete owner request records, rendered prompts, schemas, separate grading labels, code hashes and policy in an exclusive new directory. Never send expected replies, expected decisions or grading rationale to the provider. Check every exact rendered raw prompt with the bound native tokenizer before generation. Reject overflow; do not truncate. Preserve the recorded claim order and complete two-candidate cohort.

Use the retained local Qwen3:14b Q4_K_M model and isolated Ollama runtime, context 8192, output limit 128, temperature 0, seed 2026110807, raw ChatML input with thinking disabled. There are exactly 16 planned serial calls and no semantic warmup, retries or follow-up prompts. Each call has a cooperative 15-second deadline. Stop the run if provider completion is uncertain or any ownership, runtime, source fence or journal check fails. A confirmed completed malformed reply remains a protocol failure and does not get repaired or retried. Keep the unattempted suffix explicit.

The runner journals the full immutable owner request before each call, then the full provider reply before decoding. Use the existing exclusive flush-and-fsync journal. The execution wrapper must verify model/runtime identities and token counts, preserve ownership proof and perform verified cleanup. A completed runner summary alone does not prove cleanup or a successful experiment.

Grade the exact temporal header and each claim's two-letter judgment against the frozen labels. Report exact whole-reply correctness, per-claim association, per-case decision and selected fact IDs. Report malformed replies and uncertain/unattempted calls separately. Do not count unavailable results as successful abstention. Preserve both orders in paired cases; a correct result in one order cannot compensate for a wrong result in the other. Source-instruction resistance is only one case here, not a general security claim.

Report provider wall time, provider-reported input/output tokens and timings, and runner processing time separately. Runner processing includes response journaling and decoding but excludes request journaling, initial retrieval, source annotation, startup and cleanup. Report the first generation separately; do not call it warm. No timing from this probe satisfies the full warm 10k retrieval gate. Compare actual generated token counts with the earlier synthetic tokenizer estimates without treating those estimates as latency measurements.

Preparation and runner checks passed 40 tests in 2.11 seconds (`.venv/packed-probe-run-v1.xml`). They cover reordered inputs, separate grading artifacts, contradictory expectations, durable reply retention, malformed completion, uncertain completion, cancellation and journal failure. These are model-free checks. The next step is a verified isolated tokenizer preflight and execution wrapper; no model response has been collected for this probe.

The prepared artifacts are retained in `.venv/packed-query-probe-development-v1`. Their SHA-256 digests are:

- Execution inputs: `d9e7c06735db68b22783511b9cbba76beedc462c4eeaad3336d01938e9cb85a3`
- Offline grading labels: `59ddedbad6e1a9e8070bf5ccf57be950cc5d6ba6262e805dca58ccd6c0569955`
- Policy and identities: `1356b3272b4e89aaeab1521cec8f2779635234eaee27194fca49e053335a397d`

The source identity explicitly identifies authored annotations rather than model extraction. All 16 owner records and raw prompts are saved. These files do not yet bind a native process, tokenizer result or completed execution wrapper; those must be frozen and verified before inference.
