# Referent relation and consistency diagnostic v6

Status: prospective. Freeze source, corpus and protocol before inference.

The v5 comparator scored 19/20 final decisions while a post-run review found
seven incorrect intermediate relations. Six still produced the expected review
operation. This study scores those relations explicitly and checks whether
presentation changes alter the same underlying judgment. It does not modify
the comparator prompt or production defaults.

Use `benchmarks/local_memory/fixtures/referent-stability-development-v1.json`,
SHA-256 `d69661dafbe6c8099dd5a110fcecdce5f2848d1a868f4fe964d42aaf831b21c5`.
The corpus contains twelve authored seed scenarios in five clusters, each with
six matched views: base, renamed IDs, added distractors, reversed augmented
order, renamed augmented IDs and renamed reversed augmented order. The 72
views contain 198 candidate-target relations. They are not 72 independent
scenarios and are development diagnostics, not fresh held-out acceptance.

Each view preserves its source event and supplied classification. Logical
target identities bind the same literal content, version and expected relation
across views. Two complete unrelated distractors are added in the augmented
views. Expected operations and all intermediate relations are labeled before
observing model output. Ambiguous prospective requirements must remain review
outcomes rather than becoming unsupported additions. Include polarity changes,
paraphrased properties, missing qualifiers and incompatible qualifiers.

Run the unchanged `referent_v5` treatment once over all 72 views. Use a full
schedule shuffle with seed 2026110701. Require exact source and target snapshots,
raw prompts, schemas, responses and token accounting. Expected labels and
logical target aliases must never be sent to the provider.

Report operation correctness, wrong accepted targets, wrong accepted additions,
operational failures and correct reviews separately from relation accuracy.
Every target must occur exactly once with its observed version. Missing,
duplicate, unobserved or malformed relations fail the relation score. A runtime
failure receives no successful relation or abstention credit. Report combined
exactness only when both the final operation and every relation are correct.

For each seed, compare final operations after mapping IDs back to logical
targets, and compare the relation assigned to every logical target wherever it
appears. Missing, duplicate or failed views remain in the denominator and cannot
pass stability. Keep operation stability and relation stability separate.
Consistency does not imply correctness: an identically wrong relation remains
wrong in every view. Report all-six-view correctness by seed as well as totals.

Use the existing local Qwen3:14b Q4_K_M model, digest
`bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`,
Ollama 0.40.9, generate route, thinking false, temperature zero, context 8192
and one request slot. Retain the v5 budgets: 4096 output tokens, a 3584 counted
input-token cap, a 512-token template reserve, a 90-second provider deadline
and a 120-second observation deadline. Count exact prompt plus schema using the
verified native tokenizer before inference. Preserve any oversize-input failure.
The owned runtime has a three-hour bound including cleanup.

Before running, check the generated fixture against its authoring source,
mechanically exercise all 72 labeled views and 198 relations, and verify that
the scorer detects an incorrect relation hidden by a correct review operation.
Independently rebuild all recorded scores and consistency groups after the run.
Inspect literal value spans separately from semantic accuracy. Bind every source
and artifact hash, preserve failures and stop reasons, unload the model and
verify all owned processes and endpoints are gone.

This diagnostic cannot qualify a candidate for production or replace the failed
v5 confirmation gate. It supplies prospective relation and consistency evidence
for the next model or contract comparison. Keep the earlier scores and the
post-run v5 review unchanged. A separately prepared Qwen3.5 model is not part
of this run. No database writes, extraction-quality inference, retrieval,
embedding, native-host use, live configuration, pilot days or publication occur.
