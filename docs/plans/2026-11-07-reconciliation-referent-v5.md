# Reconciliation referent development v5

Status: prospective. Freeze source, input data and protocol before inference.

The identity v2 comparator generated invalid quotations and sometimes treated
a changed value as an identifying qualifier. The decoder v4 screen left the
three direct-prompt target errors unchanged. This study tests a different
comparison contract: select literal value spans by word ID, then decide whether
the complete statements identify the same variable, different variables, or
insufficient identity. A selected span establishes its source location only.
Semantic correctness remains an evaluated outcome.

Reuse the 20-case preclassified development corpus at
`benchmarks/local_memory/fixtures/reconciliation-development-v1.json`, SHA-256
`366ae35e01f517b88a0826dc2f109d51775065d552fcd0eb992033196776fef0`.
Run these three arms once per case, 60 observations, shuffled with seed
2026110701:

| Arm | Reconciliation |
| --- | --- |
| direct_full | Existing direct/full relationship prompt |
| identity_v1 | Existing subject/attribute/qualifier comparisons with quotations |
| referent_v5 | Value-span evidence followed by variable-identity comparison |

The two identity arms share the same routing and decision rules. Only direct
user revisions enter identity comparison. Other kinds retain the legacy
relationship path. Empty target sets and duplicate exact user facts retain
their existing mechanical outcomes. A correction requires exactly one complete
match and every other target ruled out; a second possible target requires
review. A prospective requirement with all targets ruled out can be added;
a correction without a referent cannot. Source role, scope, provenance,
current-version and duplicate-identity guards remain authoritative.

The new response selects one candidate value span and one value span per
target, or null when no value can be located. Inventories use non-whitespace
words with literal character offsets, including punctuation. Inclusive start
and end IDs must be ordered and present in their own source. Each target ID
and version must occur exactly once. Malformed output, invalid spans, runtime
rejections, timeouts and interruptions are failures, with no abstention credit.
Spans do not authorize a revision. The semantic relation still comes from the
model and can be wrong despite valid evidence locations.

The instructions distinguish the question a fact answers from its assigned
value. Changes to a schedule, number or required/not-required status need not
change the question; a missing or incompatible identifying condition can.
They contain no corpus-specific entities or expected answers. Instructions,
evidence representation and relation labels change together. This compares
complete treatments and does not isolate the causal effect of span IDs alone.

Before inference, verify all expected alternatives can pass each arm, compare
all source input snapshots and direct prompts with the previous study, and
replay the earlier identity outputs to confirm that the refactored shared
routing preserves their original decisions and failures. Keep the original
records and scores unchanged.

Use local Qwen3:14b Q4_K_M, digest
`bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`,
Ollama 0.40.9, generate route, thinking false, temperature zero, context 8192
and one request slot. Each arm gets 4096 output tokens, 3584 counted input
tokens, a 512-token template reserve, a 90-second provider deadline and a
120-second observation deadline. Count exact prompt plus outbound schema.
The owned runtime has a three-hour limit including cleanup.

Preserve all attempts, raw responses, prompts, schemas, native counts, provider
token accounting and timings. Independently reconstruct exact outcomes,
correct abstentions, wrong targets, wrong additions and operational failures.
Audit the selected spans against the original text. Structural validity must
not be reported as semantic accuracy. Preserve all negative results.

The referent treatment qualifies for a separately frozen three-repetition
confirmation only if all 20 cases pass with zero wrong targets, wrong additions
or operational failures. A clean screen is still development evidence; full
extraction, persisted outcomes and fresh evaluation remain required. No source,
prompt, label or limit changes are allowed while inference runs.

Verify model and source identities, unload the model, stop only owned processes
and check their absence and closed endpoints. Preserve cleanup warnings. No
database write, retrieval, embedding, native host, live configuration, installed
pilot artifact or publication is involved.
