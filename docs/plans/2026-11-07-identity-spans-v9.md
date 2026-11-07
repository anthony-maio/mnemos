# Literal identity-dimension evidence v9

Status: prospective. Freeze source, fixture, grading and protocol before inference.

The v8 pairwise path removed storage IDs and competing targets from each model
request. Qwen3.5 still made six wrong corrections for one requirement with an
unspecified environment. Its value spans sometimes covered almost the complete
statement, including the identifying environment. Test a different evidence
contract that separates identity dimensions from assigned values.

For each isolated literal pair, request subject, attribute and qualifiers. Each
dimension supplies same/different/unspecified and a nullable word span on each
side. Runtime checks require ordered in-range spans and literal evidence on both
sides for a same/different subject or attribute. Same qualifiers need both spans
or neither; different qualifiers need both. One absent side requires unspecified.
Invalid claims remain operational failures, not silently converted reviews.
Derive the overall relation: any different dimension is different_variable;
three same dimensions are same_variable; otherwise use insufficient_identity.
The runtime retains actual IDs/versions and the existing operation/scope guards.

These checks cannot detect a model assigning the wrong semantic role to literal
words or omitting a qualifier that is present. Retain all six span fields and
three dimension judgments. Score the resulting relation against every existing
label and report structurally rejected responses separately. Do not describe
literal grounding as proof of semantic correctness. The comparator is opt-in;
existing quoted, batched and v8 pairwise behavior must remain unchanged.

Use Qwen3.5:9b-q4_K_M, the stronger v8 candidate, with manifest
`56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`
and model blob
`02d45dc1cf451ba2475ac33b301c2dd8f985abe4c182ce04a1f2f5bf0260278d`.
Keep Ollama 0.40.9, generate route, thinking false, temperature zero, top_k 20,
top_p 0.95, min_p 0, repeat_penalty 1, presence_penalty 0, frequency_penalty 0,
seed 2026110701, context 8192, output allowance 4096, counted input cap 3584,
template reserve 512, provider deadline 90 seconds and case deadline 120 seconds.
Bind actual loaded model, tokenizer, context and one request slot. Keep one
structured smoke check and five tokenizer repeat/roundtrip probes separate from
quality observations. Count every planned input with the native tokenizer first.

Run the unchanged `referent-stability-development-v1.json`, SHA-256
`d69661dafbe6c8099dd5a110fcecdce5f2848d1a868f4fe964d42aaf831b21c5`,
once, using the existing shuffled schedule seed. There are 72 planned cases,
198 annotated relations and up to 198 pair requests. A failed request stops that
case, so later pairs in the failed case may remain unrequested. They receive no
credit and remain in the denominator. Do not retry a completed observation.
Automatic provider retries remain visible. There is no response cache across
cases. Expected labels and real storage identities stay out of model requests.

Verify all 198 prospective requests across 34 distinct literal pairs. ID aliases,
target order and distractors must leave each pair's request unchanged. Compare
the existing 198 v8 prompts and schemas with their retained originals before
inference. Test missing witnesses, one-sided conditions, malformed spans, source
and version binding, later-pair failure, relation grading and runner capture.

Compare descriptively with the retained Qwen3.5 v8 observations, not an
interleaved control. Report operation/relation/combined accuracy, review outcomes,
wrong targets/additions, failures, stable/correct seed groups, actual model calls,
input/output counts and latency. Retain and independently audit every request,
response, span, dimension, version binding, snapshot and original score. Replay
responses through runtime validation. No model calls occur during the audit.

Require all 72 operations and all 198 relations correct, no unsafe acceptance
and no operational failures before considering separate original-screen and
three-repetition confirmation. The original 20-case, fresh-quality, extraction,
persisted-memory and daily-use obligations remain. No promotion follows from
improved averages alone. Archive private evidence, unload the model and verify
owned processes and ports are stopped. No live configuration or publication occurs.
