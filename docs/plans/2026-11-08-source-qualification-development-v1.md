# Revision-bound source qualification development

Status: implementation contract, not a validated treatment or production change. The binary token-support run remains closed. Its fixed threshold and original scores are unchanged.

## Hypothesis and limits

The next mechanism separates query-independent fact/source qualification from query-time answer selection. Earlier checked-v5 asked several questions during one query-time judgment; this proposal instead makes source qualification an explicit revision-bound input whose reuse and invalidation can be tested separately. This is a combined architectural treatment, not an isolated prompt experiment.

A qualification records whether one retained source establishes the complete fact, including scope, quantity, time, conditions, negation, quotation, and uncertainty. It is a fallible model assessment, never proof or a new authoritative fact. An exact quote alone cannot establish entailment. Partial source coverage must not authorize the original broader fact or silently rewrite it.

Query-time selection must separately identify the requested relation and its source-backed answer. An event, date, or replacement is insufficient for a request for its cause. Keep every eligible original candidate and source visible, including rejected and unresolved qualifications. Qualification annotations must not remove comparison evidence or allow an unsupported peer claim to override a supported fact. Source conflicts still require query-time resolution against the full original snapshot.

## First implementation boundary

Implement an in-memory development-only qualification record and lookup. Do not add a persistent cache, dependency, service, database migration, or production retrieval mode.

Bind each record to exact fact ID, version, content, scope, scope ID, own evidence ID, excerpt, source event ID, source reference, and qualification policy/model identity. Use canonical serialization and a collision-resistant digest; reject malformed or incomplete identity. A different policy or runtime must miss the lookup. Do not key by text alone or by fact ID alone.

Statuses are supported, unsupported, and unresolved. Provider failure is unavailable and must not be cached as unsupported or counted as successful abstention. Store explicit assessed results only. A record for one source cannot qualify a different source of the same fact. A missing record is unresolved, not unsupported. Preserve original candidates and source order in any annotated view.

The caller must check the complete live snapshot before and after provider work. Exact record matching alone does not establish current authorization or protect against forgotten facts. Scope eligibility and forget suppressions remain authoritative at lookup. The experimental record layer must accept only currently eligible candidates supplied by that caller; it must not enumerate historical records or restore a fact to the cohort.

Required model-free checks: correction invalidates; excerpt and source-reference changes invalidate; policy and model changes invalidate; same text across scopes does not collide; own-source binding holds; unavailable results cannot become negative records; an omitted or forgotten candidate is never reintroduced; rejected and unresolved candidates remain in the annotated original cohort. These checks establish control flow only.

## Evaluation before any promotion

Before new inference, freeze a separate authored development set, source-qualification labels, answer labels, exact prompts, runtime identities, fixed decision rules, and request schedule. Include complete and partial claims, valid proposed/quoted/negative facts, unsupported replacements, explicit conflicting sources, missing causes, and supported causes. Do not reuse the closed binary cohort to claim fresh quality. Keep all labels outside inference.

Measure source qualification separately from query-time answer support. Report false qualification, false rejection, unresolved and unavailable cases, and the original answer precision, critical retention, recall, and abstention. Compare treatments on identical immutable accepted-fact snapshots. A qualification pass does not establish extraction quality.

Measure cold qualification cost, warm lookup cost, misses after correction/forgetting, whole-query support time, and ultimately complete retrieval at 10,000 facts. Do not exclude qualification from capture acknowledgment if capture actually waits for it. Asynchronous qualification must expose queued versus completed state and must not serve an unchecked cached answer while pending. A warm-only result cannot establish first-use latency or the 100 ms durable capture gate.

Keep the existing 95% precision and critical retention, 90% recall and abstention, 500 ms complete retrieval, and 100 ms durable capture requirements. A fresh acceptance cohort and the original two-project actual-use pilot remain separate obligations. No implementation or synthetic check activates this mechanism.

## Prospective selector rules

The first adapter uses one binary scored token per source qualification and per answer witness. Require both offered labels in the returned alternatives. At reported probability at least 0.9, generated A means supported and generated B means unsupported; otherwise the assessment is unresolved. These are uncalibrated decisions. A selected witness must pass both stages. No thresholds may change after the new evaluation starts.

Source qualification receives only the complete fact and its designated source, without the query or reference date. Query-time judgments receive the explicit reference date, query, complete original cohort, all qualification annotations, and one designated fact/source. Every witness receives an answer judgment, including unsupported and unresolved qualifications; such qualifications cannot pass the final conjunction. No original comparison evidence is removed.

The complete cold call set is capped at 64 requests and 128,000 UTF-8 prompt bytes, checked before the first provider call with the longest possible annotations. Role delimiters are rejected during preflight. The provider must still check exact model token counts without truncation. A caller-owned deadline and live revision fence surround both stages. Source results are committed only after the entire missing-source batch succeeds; a failure cannot become a negative qualification. Valid source records may remain after a later answer failure, but no partial answer is returned.

Source policy identity binds the source prompt and fixed rule. Model and runtime identities are separately mandatory, and the execution freeze must bind actual adapter, decoder, prompt frame, and provider configuration. The caller is responsible for those identities matching the actual provider. This adapter has no provider installation or identity attestation of its own. Any prompt, decoding, or qualification-rule revision requires a new policy identity before reusing records.

## Development v1 cohort and execution order

The separate qualified-support-development-v1 fixture contains 12 scenarios, 25 fact/source pairs, and 24 questions: 12 answerable and 12 absent. Source labels contain 14 supported and 11 unsupported pairs. Labels are authored development judgments, not independent annotation. They remain in a separate file from runtime inputs. This set includes both supported and missing causes, partial coverage, accurate proposals and quotations, negative findings, unsupported replacements, required versus measured values, future effective dates, conflicting sources, implied regular intervals, and source instructions.

For each question in lexical key order, run the unchanged token-support baseline, the new selector with an empty qualification cache, and the new selector with only that question's completed cold qualification records. Each question gets its own cache. Warm repeats are timing/reuse observations, not additional independent semantic cases. Expect 200 semantic calls in total without failures: 50 baseline, 50 cold source, 50 cold answer, and 50 warm answer. Use one unrelated warmup before the sequence. If the cold source batch fails, record the dependent warm arm as unavailable without retrying source qualification. A completed source batch may be reused after a cold answer failure, which must remain reported.

The fixed order favors later arms through provider prompt caching and other runtime warming. Report that limitation; do not interpret a warm-arm difference as an isolated causal benefit of the qualification cache. Preserve per-stage calls, prompt counts, warm/cache-hit indicators, failures, and whole-question times. The baseline and treatment must use exactly the same accepted candidates, retained sources, scope, reference date, and query. Retrieval and extraction are excluded from this experiment.

The model-free preparer enumerates all three source dispositions for every witness, without using gold to choose annotations. For this one-source-per-fact cohort, the resulting prompt universe contains 633 unique frames: 25 source, 50 baseline, and 558 annotated answer frames. Before inference, tokenize every frame exactly and retain token IDs. Runtime requests must match one frozen frame and its token budget; never truncate. Final runtime identities, model settings, ownership checks, and cleanup still require an execution freeze. No semantic inference has been authorized by a passing structural preflight alone.
