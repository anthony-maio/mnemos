# Isolated extraction development v1

Status: prospective development measurement. Commit the corpus, scorer, runner
and this protocol before inference. Preserve every attempted case and raw reply.
No extraction prompt, threshold or production implementation changes are part
of this comparison.

Run the current `source_bound` classification and relationship reconciliation
functions on 16 authored development cases. These reuse previously studied
scenario families with new fixture text. They are not fresh evaluation data.
Fifteen cases measure extraction outcomes; one deliberately foreign target
measures pre-provider rejection and stays outside quality denominators. Four
quality cases require no retained outcome. Host-channel labels are synthetic,
not evidence of native-host execution.

Use the fixture `benchmarks/local_memory/fixtures/extraction-isolated-development-v1.json`.
Bind its exact SHA-256 in the command and execution manifest. Freeze each event's
identity, role, time, scope, profile and source channel, each target's identity
and version, and all expected source spans, kinds and operations. Labels are
available only to the local scorer, never to the model. Use shuffle seed
20261107 and one repetition for this development check.

Call the same source pipeline and relationship adapter used by the production
source-bound service path. Keep corrections enabled at the proposal boundary.
This measures proposals and explicit reviews, not committed facts. Do not query
retrieval, compute embeddings, run other modules, mutate accepted stores or
change live host settings. The two-stage source pipeline returns exact source
slices and guarded operations; no alternative model prompt is introduced.

Use existing Qwen3 14B Q4_K_M model digest
`bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`,
installed Ollama 0.40.9, one private loopback runtime, context 8,192, temperature
zero, thinking disabled and output limit 2,048. Check each prompt plus decoder
schema with the native literal tokenizer before inference; cap that combined
count at 5,632, leaving 512 tokens for template overhead and 2,048 for output.
Keep the provider timeout at 30 seconds and the whole-case deadline at 60.
Record logical calls, completions and all observed raw responses separately,
since the existing provider can retry transient transport failures. No model
download, cloud fallback or unrelated application shutdown is authorized here.

Grade exact annotated outcomes only when source identity, role, channel, profile,
scope, content hash, character span and text match the retained event. Require
the expected kind, operation, target and observed version. Check that an accepted
staged outcome actually has its corresponding proposal. Matching is one to one:
duplicates cannot increase retained-fact counts. An unresolved review cannot
satisfy a critical accepted-fact expectation.

Report critical exact retention, accepted-outcome exact precision, unmatched
accepted outcomes, review outcomes, exact case matches, valid abstentions and
operational failures. Also report critical-span containment separately; a fact
with an extra temporary instruction is not an exact supported outcome. These
are deterministic annotation proxies, not semantic factual-precision estimates.
Failed empty operations receive no abstention credit. The foreign-scope guard
passes only for the expected error with zero provider calls.

Retain full classification inventories, candidates, reconciliations, proposals,
reviews, actual input snapshots, prompts, schemas, native token IDs, responses,
errors and latency. Independently reconstruct source bindings and scores from
those records. Unload the model and stop only the owned runtime afterward.

Before a larger prospective extraction evaluation, review the development
result, freeze the chosen candidate and author a separate corpus with the
existing sample and repetition requirements. Semantic review, actual fact
commits, correction/forget integration, native hosts and pilot acceptance remain
separate. A successful development run cannot close those requirements.
