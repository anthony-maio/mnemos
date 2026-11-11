# Optional Jev support comparison

The optional Jev backend completed all 44 known hybrid-arm development
questions. It selected 21 supported facts and no unsupported facts, and
abstained on every no-answer question. It missed six answerable questions.
This configuration fails the original retention and recall gates. Local
operation remains the required direction; no defaults, live configuration,
pilot installations or memory databases changed.

| Metric | Jev 1.14.0 | Previous retained Qwen3.5 configuration |
| --- | --- | --- |
| Expected facts retained | 21/27 (77.78%) | 27/27 (100%) |
| Selected-fact precision | 21/21 (100%) | 27/41 (65.85%) |
| Answerable-question recall | 21/27 (77.78%) | 27/27 (100%) |
| Correct no-answer abstentions | 17/17 (100%) | 9/17 (52.94%) |
| Support-only p95 | 441.73 ms | 2,416.00 ms |
| Unavailable / unattempted | 0 / 0 | 0 / 0 |

These are 44 known questions, not fresh acceptance data. Both runs use the
same frozen hybrid-arm question, candidate, full-source and reference-date
payloads, and the same 0.8 selection rule. The prior local results were read
from saved evidence; no local inference was repeated. There was no new
extraction, retrieval or accepted-store regeneration. Critical retention is
unmeasured because these labels lack a critical-fact registry. The 95%
precision/retention and 90% recall/abstention gates remain unchanged. Support
timing excludes retrieval and cannot establish the complete 500 ms recall gate.

Source `85218825ea158268d0c446bd87e81ad90545553d` ran one unrelated warmup,
then one sequential pass. The warmup resolved `jev-latest` to `jev-1.14.0`;
all measured requests explicitly used and returned that version string. This
is vendor-reported identity, not a reproducible local weight hash. Execution
took 18.09 seconds. Each operation had an absolute 15-second deadline, with
720 seconds overall, no retries and no fallback. HTTPS certificate verification
remained enabled; redirects and proxy environment settings were disabled.

The replay required complete finite normalized raw probability distributions
and a maximum-probability selection before the unchanged grader ran. Exact
wire bodies and raw responses were retained without authentication headers.
The credential was read only inside the isolated process from the existing
private `.env`; it was not exported, copied to tracked configuration or printed.
Pre/post hashes matched, and the credential scan found no key in the evidence.
The process finished and no server was launched.

Reported usage, including warmup, was 38,037 input tokens and 5,260 output
tokens. At the provider's [published input price](https://help.typesafe.ai/articles/5187202169-how-much-does-jev-cost),
the estimated cost is $0.001597554. This is a usage estimate, not a billing
receipt. The [API contract](https://docs.typesafe.ai/api/system-one) and
[model listing](https://docs.typesafe.ai/api/list-models) were checked on
November 11. Calibration on this task is unmeasured.

Validation: 33 focused tests passed across hosted replay, frozen-request
restoration/grading, TypeSafe behavior and backend settings. A separate
read-only review checked execution bounds, raw probabilities, model identity
and credential handling before the run. No product runtime source changed.

A separate post-run audit matched all 44 inputs and labels to the historical
hybrid subset, checked all 45 wire request/response pairs including warmup,
validated every raw distribution and model identity, and independently
recomputed the grades, p95 and usage. All 95 frozen file hashes matched.
It found no material execution or scoring issue. Credential absence relies
on the guarded writer and execution receipt; the reviewer did not access the key.

The [machine-readable summary](jev-support-screen-v1.json) preserves all
denominators and missing requirements. Private raw evidence is retained in
`.venv/jev-support-screen-v1`, with a portable archive at
`.venv/jev-support-screen-v1-portable-evidence.zip` (105 files including its
manifest, 456,779 bytes), SHA-256
`b26fcb8aee817abcef083daadfdbc3c10fcf34a1ad8841874fcd1ea73895aa85`.
The frozen input/source manifest SHA-256 is
`a775bfae9c307a584bbf436b217eba2152f6cb410810a89811c4e42243dccde0`.

Keep Jev optional and preserve this failed gate result. Do not lower the
threshold to rescue recall or treat this as a local-support solution. Continue
with the local workflow requirements, not another support-model sweep.
