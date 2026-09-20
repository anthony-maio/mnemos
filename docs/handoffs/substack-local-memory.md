# Mnemos local memory: research and Substack writing handoff

Status: working handoff, September 19, 2026. Engineering and validation remain in progress. Reconcile the final checkpoint and open results before treating this as a finished account.

## Assignment

Write approximately 2,500 words for Anthony Maio's Substack about the work of turning Mnemos's memory promises into a dependable local system. Explain what we found, what changed, what the experiments actually establish, and where the evidence stops. The reader should understand why remembering a fact becomes difficult when several agents can capture it, correct it, forget it, or lose contact with the service.

Apply [writing-in-anthonys-voice](C:/Users/Ant/.agents/skills/writing-in-anthonys-voice/SKILL.md), the skill Anthony explicitly selected. Read its embedded examples before drafting. Use connected explanatory prose, concrete examples, straight quotes, and standard US keyboard characters. Do not plant typos, manufacture personal reactions, or invent experiences. This is an article, so use the skill's article guidance rather than its quick social-post style. If transferring to another machine, include the skill file with this handoff.

Deliver an article draft, three possible titles, and a separate compact source/claim note. Keep the article close to 2,500 words, excluding source notes. Do not publish it.

## Latest native result and portable primary sources

The [fixed Hermes v4 cohort](../validation/2026-09-19/hermes-native-cohort-v4/README.md)
ran all three planned attempts on verified runtime `8f8f63f`. All three teach
phases saved the exact fact and called job status. Full lifecycles still
finished zero for three. Attempt 1 passed correction, restarted recall,
forgetting, and known-source replay, then retrieved the deleted value from
newly captured assistant narration. Attempt 2 failed on locale-encoded bridge
output. Attempt 3 retrieved and inspected a fact, then announced a correction
without calling the correction tool. Preserve the difference between a model
claim and an observed mutation. The three-host driver remains unrun.

The earlier [v3 cohort](../validation/2026-09-19/hermes-native-cohort-v3/README.md)
also failed all three attempts, at teach, because each model omitted the
job-status call. Its results remain separate. V4's later progress is useful
evidence, but is not a completed integration or a controlled estimate of the
guidance change's effect.

The packet now includes [dated website and paper snapshots](../validation/2026-09-19/writer-primary-sources-v1/README.md),
with original URLs, retrieval timestamps, response metadata, and hashes. These
preserve the current public sources, not a proven snapshot of the original
audit date. Cite the public URLs and distinguish their promises from later
local validation. The paper's broader competitor and neuroscience claims
still need independent primary-source research before repeating them.

## Starting point and source map

The initial question was the gap between [the Mnemos website](https://mnemos.making-minds.ai/) and [the paper](https://making-minds.ai/papers/mnemos.pdf), and what it would take to make the software functional local memory. Read those primary sources directly and record the versions or retrieval dates. Do not treat the current website as an immutable record of the original audit.

Local checkout: `D:/Development/mnemos/mnemos/.worktrees/functional-local-memory`, branch `wt/functional-local-memory`. Repository-relative links below resolve within that checkout. A writer without repository access will need the linked documents and relevant evidence artifacts exported alongside this handoff.

| Question | Starting evidence |
| --- | --- |
| What was promised and what was missing? | [Original audit and T01-T19 plan](../plans/2026-09-16-functional-local-memory.md), especially capability inventory and paper reconciliation |
| What is implemented and what remains open? | [Implementation status](../validation/2026-09-16/implementation-status.md), [claim/evidence ledger](../validation/2026-09-16/claim-evidence-ledger.md), and [progress](../validation/2026-09-16/progress.md) |
| How does the memory service work? | [Functional local memory](../functional-local-memory.md) and implementation at the cited commit |
| Which host claims have native evidence? | [Client compatibility matrix](../client-compatibility-matrix.md) and its linked native-host records |
| What did the comparisons show? | [Evaluation status](../validation/2026-09-16/evaluation-running.md), [comparison diagnostics](../validation/2026-09-19/comparison-diagnostics.md), and raw reports linked from them |
| What should change in the research claims? | [Paper errata](../paper-errata.md) and the claim/evidence ledger |

Status prose can lag newer artifacts. Resolve disagreements against the commit-bound test reports and raw evaluation outputs, and retain the discrepancy in research notes until corrected.

## Suggested narrative, about 2,500 words

1. **The practical question, 250 words.** Several coding assistants should share useful memory across sessions and projects. Introduce one concrete failure that explains why that is harder than saving text. Use a verified fixture example and label it as such if no observed daily-use example exists.
2. **The promise and the audit, 400 words.** Explain the five modules in plain language: deciding what is surprising enough to retain, updating memories from evidence, using affective state in retrieval, background consolidation, and associative graph retrieval. Separate biological inspiration from measured software behavior.
3. **What dependable memory required, 700 words.** Follow one fact through capture, durable acknowledgement, extraction, scoped retrieval, correction, restart, and forgetting. Use a few actual repairs to make the consequences tangible: scope isolation, stable event identities and retries, revisions and derived graph state, write checks, and suppression of forgotten content.
4. **What the experiments exposed, 650 words.** Explain why passing deterministic tests does not prove extraction quality or better recall. Include the real-model comparisons, negative results, and the difference between a faster client fixture and faster end-to-end memory. State incomplete experiments plainly.
5. **What can be claimed now, 350 words.** Give the final observed state of Codex, Claude Code, and Hermes integration, plus the pilot status. Explain practical operating choices only where the evidence supports them.
6. **Where this leaves the work, 150 words.** End at the actual engineering result. An unresolved research question is acceptable. Do not invent a successful release or a daily-use experience to produce a neat ending.

The hardware interruption is available as context: Anthony reported that the remote dual-3090 machine failed and moved the work to a local RTX 4080 with 16 GB. Treat the failure history as user-reported; do not infer an electrical cause. Colab was considered and tabled. Include this only if it helps explain the experimental constraints.

## Capture choices worth explaining

Anthony asked for advice about automatic conversation capture versus saving
only facts an agent explicitly selects. Present this as an engineering
tradeoff, not a settled personal preference or a proven deployment decision.
Automatic capture preserves source context and reduces omissions, but adds
noise, extraction cost, sensitive material, and opportunities to recapture
stale assistant claims. Explicit selection makes intent clearer and reduces
volume, but depends on the agent noticing what matters and actually calling
the tool. The native v4 results illustrate both problems directly.

The proposed operating approach is a hybrid: opt-in local capture as source
material, conservative promotion into durable facts, explicit remember and
correct actions when intent is clear, source-backed retrieval, and deletion
that covers known copies and replay. Captured conversation is not itself an
asserted fact. Describe this as the design direction; the open capture,
quality, and lifecycle gates prevent a claim that daily operation is already
dependable. Do not turn automatic capture into permission to retain every
project or all private conversation indefinitely.

## Current evaluation update

Native evidence has a further qualification: the new [offline trace checks](../validation/2026-09-19/three-host-runner-offline-v1/README.md)
read six historical Codex/Claude sessions. Claude exposes directly correlated
tool-use/results. Codex's code-mode cells and MCP completions use different
identities, so the parser retains operation observations without claiming a
direct model-call match. The prospective three-host test still needs this
correlation gap resolved. Passing the parser's offline tests is not a new
native lifecycle result.

The independent encoding comparison ended on September 19 at 22:24:29 UTC
with all three reports marked incomplete. Of 600 observations per arm, the
full pipeline has 11 partial rows, no-surprisal has eight, and length-noise
has 15. Preserve these 34 partial rows out of 1,800 when discussing results;
metrics that exclude them are conditional diagnostics. Retrieval began next
and finished with incomplete coverage. The [terminal encoding archive](../validation/2026-09-19/encoding-terminal-local-v1/README.md)
preserves the original reports and explains denominators and conditional
paired differences. The subsequent [terminal retrieval archive](../validation/2026-09-19/retrieval-terminal-local-v1/README.md)
contains 589 complete and 11 partial observations, 900 query rows, and 3,600
strategy results. Its three alternative strategies have zero paired quality
delta against full on eligible rows. Neither archive closes complete-run
quality acceptance.

## Latest verified repair

The [native failure repair checkpoint](../validation/2026-09-19/native-repairs-checkpoint-v1/README.md)
at `fd83fb2` passes 1,351 source tests with 4 skips, all six source/package
checks, 569 installed cases with 4 matching skips, and seven replay
assertions. Wheel SHA-256: `4958c5ef7a509fefecacc0666d87f578b46e94e1a33a896831468c0f971bf895`.
It fixes explicit UTF-8 bridge output, typed missing-fact errors, and delayed
assistant capture that quotes a forgotten fact ID. The end-to-end regression
fails against preserved `8f8f63f` source and passes after the repair, including
restart, separate scope controls, and explicit reteaching with a new ID.

The assistant-ID safeguard conservatively suppresses an entire automatic
assistant event when it mixes an erased ID with unrelated material. It covers
exact token IDs, including native UUIDs; it does not establish arbitrary
paraphrase erasure or deletion of every historical raw source. User captures
and explicit writes remain eligible. The three earlier native v4 failures are
unchanged, and this repaired package has not had a new native-model run.

## Package used by native v4

The [affective parsing and write-guidance checkpoint](../validation/2026-09-19/affective-guidance-checkpoint-v1/README.md)
at `8f8f63f` passes 1,324 source tests with 4 skips, all six source/package
checks, 543 installed cases with 2 matching skips, and the installed replay
probe. Its wheel SHA-256 is
`53444f7f87141e7107fe9f4daf4a6095e04c9d69d007cd218634fd2fff50f99f`.
This is the package used by native v4. The parser accepts signed decimals and
scientific notation, rejects malformed or nonfinite triplets, and preserves
prior state on failure. Replaying 300 archived responses changed zero states,
so the original 33/300 classifier-fidelity result remains negative. The
[fidelity review](../validation/2026-09-19/affective-fidelity-review.md) also
explains that the gold vectors were authored synthetic labels, with no
independent human calibration in the archive.

## Earlier verified repairs

The [job lookup error checkpoint](../validation/2026-09-19/job-not-found-checkpoint-v1/README.md)
at `68693f8` passes 1,308 source tests with 4 skips, all six
source/package checks, 533 installed cases with 2 matching skips,
and 7 replay assertions. Missing and out-of-scope jobs return the same
`404/job_not_found` response. Unexpected internal lookup errors remain 500,
and direct callers retain the `KeyError` contract. This repair leaves the
failed native lifecycle intact: it does not substitute the right job ID for
the model or claim that its pending write finished processing.

That checkpoint's wheel SHA-256 is `2e121c1ae81717d1b9a32349bd27ea7fafc2e5562e6eadbd7d24ddecd16f5cb8`. The prior package below
remains the runtime used for both failed native lifecycle attempts; do not
attribute those runs to this newer wheel.

The preceding [Hermes native identity checkpoint](../validation/2026-09-19/hermes-native-identity-checkpoint-v1/README.md)
at `3730fd9` passes 1,247 source tests with four skips, all six source/package
checks, 530 isolated installed cases with two matching skips, and seven replay
assertions. Native persisted message IDs now distinguish a regenerated turn
from a duplicate delivery. Reusing a native ID with changed content or role
fails as an identity conflict, including after service restart. This package
retains the earlier correction retry repair.

The [actual Hermes database fixture](../validation/2026-09-19/hermes-native-rows-contract-v3/README.md)
preserves the first turn's IDs during rewind and creates new IDs for an
identical regenerated last turn. Checkpoints, normal resume, and provider
recreation leave exactly six queued occurrences. This uses actual host
persistence and callbacks, without a model, service, or interactive CLI.
ID-less transcripts, loaders that omit IDs, compaction that clones rows, and
migration from existing prefix identities remain separate limitations.

The preceding native-identity wheel has SHA-256
`3772398a8e11b92d2cffd285b0571a733a9fd63cbdf932c30519d933cf0a6dd2`.
This is a tested package checkpoint, not final project acceptance. A useful
article example is that identical words can be a replay of an old observation
or a genuinely new turn. The host already had durable IDs that distinguish
them; the adapter needed to preserve those IDs and reject conflicting payloads.

## Evidence boundaries already established

- Earlier verified package checkpoint: `5e4b057b16906c2ec7dd302ae80748ef1ac5eddc`, with 1,195 source passes and four skips, all six source/package checks, 507 isolated installed contract passes and two matching skips, plus seven replay assertions. Wheel SHA-256: `7eaba268fb91ffae7d59b222cf8addb49d8b4b4857a37552e5404422e4009625`. The [Hermes lifecycle checkpoint](../validation/2026-09-19/hermes-lifecycle-checkpoints-v1/README.md) verifies scoped explicit tools, transcript identities across provider recreation, and extraction-ineligible provisional capture. Its installed Hermes API v2 run uses fixture providers and the actual Windows/WSL bridge, with no model conversation or outage. The archive also preserves two gaps found after the earlier candidate's suite passed. Overall acceptance remains open.

- The preceding `de6cdcf1c05fdad1bb19ed8f52638eff5ed67ae4` [recovery checkpoint](../validation/2026-09-19/loopback-recovery-checkpoint-v1/README.md) had 1,160 source passes and four skips, 474 isolated installed contract passes and two matching skips, plus seven replay assertions. It preserves both preceding full-suite failures and the later passing package evidence.

- The last fully verified checkpoint before the latest repair work was `8fc79f01ab68d9c211f9be4818f9e566e271cefd`: 1,143 source tests passed, four skipped; the isolated installed wheel passed 440 selected cases, two skipped, plus seven replay assertions. See the [checkpoint](../validation/2026-09-19/retrieval-parity-checkpoint-v1/README.md). These are historical counts, not a claim about the eventual final build.
- The later `d0a344056dc9f93e10098ed405fa226095457066` run had 1,150 passes, one recovery-test failure, and four skips. A reproduced client-construction delay consumed the readiness budget. The repair is included in the verified `de6cdcf` package above; preserve the failed run when explaining how the defect was found.
- Local real-model evaluation uses Qwen3.5:9b and nomic-embed-text through Ollama. Cite exact digests, settings, hardware and source commit from the relevant run manifest; do not combine different checkpoints into one result.
- The September 19 lifecycle queue ended incomplete in five of seven configurations. Baseline and the configuration without sleep consolidation completed. Encoding/retrieval follow-on comparisons did not start in that queue. An [independent v2 queue](../validation/2026-09-19/supplemental-launch-v2/README.md) subsequently produced terminal but incomplete encoding and retrieval archives under the same frozen bindings. Completion alone does not establish that a configuration passed its quality gates.
- The [terminal lifecycle archive](../validation/2026-09-19/lifecycle-incomplete-local-v2/README.md) preserves all seven reports and their raw outputs. Fifty of 4,200 observations were partial because extraction returned no durable facts and a later correction could not resolve its target. No provider timeout caused those partial rows. Do not describe this as merely a failed benchmark runner, or retry only the failed observations until they succeed.
- The [extraction diagnosis](../validation/2026-09-19/lifecycle-extraction-diagnosis.md) identifies a more specific issue: those model responses reject "The earlier project note said ..." as historical rather than current evidence, while the fixture expects acquisition of a superseded fact. This is evidence of a temporal/assertion ambiguity. It does not prove the model was right to discard the information, and it is not permission to rewrite the held-out labels. The distinction could make a useful article example if the final account preserves those qualifications.
- Graph utility and end-to-end affective benefit remain unproved. Structural tests, rankings produced from supplied reference states, and classifier fidelity measure different things. Report their denominators and limitations separately.
- Storage-placement and HTTP client experiments isolate particular costs. Their gains cannot be presented as the same gains in complete real-model retrieval. Preserve failed latency measurements alongside successors.
- Three [baseline SSD performance repetitions](../validation/2026-09-19/performance-recovery-v1/README.md) on `de6cdcf` passed both unchanged latency gates: warm recall p95 245.214-252.222 ms, durable capture p95 28.804-36.062 ms, all 150 captures committed. Each run used 33 timed queries with real Nomic embeddings, 10,000 synthetic stored vectors, and fixture capture embeddings, with all cognition modules disabled. The first seed embedding took 6,691.178 ms, recorded separately from warm recall. This is a useful result with a specific scope, not evidence of semantic accuracy, full-module latency, or a native-host conversation.
- Native adapter tests, observed native host conversations, Codex Desktop, complete automatic capture, and a two-week pilot are separate evidence categories. Do not promote one into another.
- The [September 19 native Hermes attempts](../validation/2026-09-19/native-hermes-local-v1/README.md) completed local model turns, but failed acceptance because the installed CLI omitted the provider's session directory. A correct answer obtained through an unintended process-directory fallback is not evidence of safe scope isolation. The adapter repair closes that fallback; later patched-host runs validate explicit propagation in an isolated checkout. Built-in remove/replace notifications also lack exact Mnemos targets and remain unresolved. The diagnostic queue is in-memory only.
- The later [native lifecycle attempt](../validation/2026-09-19/hermes-native-lifecycle-local-v1/README.md) stopped at teach. Its `mnemos_remember` response reported `status: committed` with `processing_status: pending`, but the model omitted the exact target's final period and an auxiliary title-generation request lacked the required context option. These are distinct failures from the earlier scope defect. The [v2 configuration preflight](../validation/2026-09-19/hermes-native-discovery-preflight-v3/README.md) confirms that the disposable host disables the optional title model without making a model or service request; the successor lifecycle has now also failed, as described below.
- The [native Hermes v2 lifecycle](../validation/2026-09-19/hermes-native-lifecycle-local-v2/README.md) fixed both setup problems: the exact-content write returned `status: committed` with its job still pending, and all three model requests carried the required context setting. The model then altered the returned job ID when calling `mnemos_job_status`. That unknown ID produced HTTP 500 and a degraded result, so the evaluator stopped before maintenance and later phases. This does not establish a completed asserted fact or a passed lifecycle. Keep the model's argument-copy error and the service's error handling distinct; no argument was repaired on the model's behalf.
- An earlier [isolated patched-host run](../validation/2026-09-19/native-hermes-local-v3/README.md) passed explicit scope, recall, foreign-project exclusion and raw user/assistant persistence. It failed the all-captures gate because duplicate callbacks generated different timestamps for the same event ID. The outbox rejected that conflict correctly. Preserve the failure and any successor separately; a patched host is also a distinct compatibility claim from the installed version.

- The [native Hermes v4 successor](../validation/2026-09-19/native-hermes-local-v4/README.md) passes all 16 bounded checks. Four capture calls returned committed while exactly two raw conversation events persisted. The native turn took 24.592 seconds, with a separate 3.569-second cold/load preflight. It uses the proposed CLI patch in an isolated host checkout; the installed host is unchanged. Full native correction, forgetting, outage recovery, extraction quality, and daily-use readiness remain open.

The [Hermes rewind diagnostic](../validation/2026-09-19/hermes-rewind-diagnostic-v2/README.md)
provides another bounded negative result. The installed provider treats an
identical regenerated turn after a same-session rewind as the original
observation. Its outbox retains four records where six logical observations
are expected. This is a callback fixture, not an interactive `/undo` run or a
service crash test. The native-ID repair above addresses the supported
persisted-message path; the ID-less fallback still has this limitation.
Normal duplicate replay must continue deduplicating. The first diagnostic's false-positive zero-capture report is
preserved and explicitly rejected.

## Finalization checklist

The [correction retry checkpoint](../validation/2026-09-19/correction-retry-checkpoint-v1/README.md)
at `ea88dd4` passes 1,201 source tests and 513 installed cases, with four and
two respective skips, all six source/package checks, and seven replay
assertions. A correction that commits before its reply is lost can now recover
its original job after restart without overwriting a newer edit. This is a
concrete reliability result from fixture providers and actual HTTP/SQLite;
it does not change the frozen real-model experiment or its missing targets.

New evidence for reconciliation: the [Hermes API recovery probe](../validation/2026-09-19/hermes-api-recovery-v1/README.md)
passes outage buffering, required checkpoint durability, native-request replay
after explicit restoration, and forget suppression across fresh host processes
on `5e4b057`. It uses fixture providers. The [graph coverage diagnosis](../validation/2026-09-19/graph-coverage-diagnosis-v1/README.md)
finds every recorded target disconnected from the selected seed; changing a
traversal threshold cannot repair those missing paths. Both records belong in
the final packet refresh, with their limitations intact.

The follow-up [development degree comparison](../validation/2026-09-19/graph-degree-development-v2/README.md)
tests four, eight, and sixteen neighbors on saved development vectors. All 15
combinations finish; sixteen neighbors restore a usable path in one case,
but no target reaches the top five. This is a concrete example of a mechanism
working without delivering the intended answer. These are five variants of
one development template, not five independent real-world workflows.

Candidate examples for the article, with bounded evidence:

| Problem | Resulting behavior | Evidence and limit |
| --- | --- | --- |
| An identical last turn is regenerated after Hermes rewind | Native row IDs preserve the surviving prefix and identify two new occurrences | [Native identity tests](../../tests/test_hermes_native_identity.py) and actual host database fixture; ID-less fallback and compaction remain limited |
| A committed capture loses its HTTP acknowledgement | The outbox retains a stable event identity for durable replay | [Lost acknowledgement tests](../../tests/test_service_lost_ack.py); synthetic service/SQLite evidence |
| A correction finishes before its acknowledgement reaches the client | An unchanged retry recovers the original job without rewriting a newer fact; a fresh edit still needs the current version | [Correction retry tests](../../tests/test_memory_service.py) and the real HTTP lost-ack test; fixture providers, with restart, raw-source expiry, scope rejection, and forget suppression covered |
| Two storage backends break exact-score ties differently | Both order tied candidates by memory ID and return no candidates for empty/nonpositive requests | [Storage contract tests](../../tests/test_storage_retrieval_contract.py) and the retrieval-parity checkpoint; consistency does not establish better semantic recall |
| Client setup consumes a short health-check budget | Plaintext loopback calls skip unused certificate loading while preserving auth and transport restrictions | [Recovery tests](../../tests/test_client_stale_descriptor.py); repeated local transport measurements, not model inference |
| A competing launcher exits before the original owner recovers | Polling can use the remaining original startup deadline without another launch | Same recovery tests; this does not prove every native host can tolerate the entire startup interval |
| A Windows reader blocks atomic publication of a new runtime descriptor | Bounded sharing-error retries preserve the old descriptor until replacement succeeds | [Publication tests](../../tests/test_service_competing_recovery.py); actual Windows held-reader reproduction and persistent-failure checks |
| Repeated Hermes callbacks reuse an ID but generate different timestamps | A private ledger retains first-observation timestamps across provider recreation; transcript prefixes distinguish repeated messages from replay | [Transcript identity regressions](../../tests/test_hermes_transcript_identity.py) and the installed API v2 record; arbitrary native transcript truncation and branching remain open |
| A callback lacks a usable transcript | The service retains a separate provisional receipt without extracting facts from it | [Provisional capture regression](../../tests/test_hermes_provisional_capture.py) and installed API v2; receipt counts cannot establish independent observations or complete capture |
| Hermes needs to correct or forget a specific fact | Six native provider tools enforce session scope, exact fact IDs and versions, and explicit write request identities | [Tool regressions](../../tests/test_hermes_tools.py) and installed API v2; full native model lifecycle and built-in memory-store reconciliation remain open |
| Another client corrects or forgets a fact after Hermes retrieves it | Each ordinary prefetch reads the current service result | Same Hermes tests include a real service/SQLite correction-and-forget check; this removes a stale query cache but does not establish every native host lifecycle |

Choose the examples that support the narrative. This table is an evidence map,
not a requirement to recount every implementation detail in the article.

- [ ] Replace provisional status with the final verified source commit, wheel identity, source/installed checks, and artifact links.
- [ ] Reconcile every evaluation configuration, including failures and incomplete runs, with the final reports.
- [x] Add a concise before/after repair table with a source or regression test for each selected article example.
- [ ] Record final host continuity, latency, capture coverage, and pilot status without collapsing the separate gates.
- [ ] Identify which quantitative claims can be used, with units, sample sizes, settings, uncertainty where available, and limitations.
- [ ] Verify original website/paper claims against primary sources and link the dated corrections.
- [x] Prepare a portable working packet with this handoff, the selected voice skill, key reports, regression examples, and a file manifest. Refresh that packet at final acceptance. The portable v8 packet includes terminal encoding/retrieval reports, the two earlier native lifecycle failures, both fixed-cohort runs (all six attempts failed), the latest repair evidence, the affective fidelity review, and dated website/paper snapshots. Some older evidence links still require the full checkout. Local validation artifacts are research inputs; public citations require accessible sources.
- [ ] Keep unverified interpretations and questions outside the article's factual claims, then make the final voice pass using Anthony's skill.
