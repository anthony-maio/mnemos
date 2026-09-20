# Native Hermes lifecycle v6: reviewed copy deletion

This prospective successor exercises two newly exposed Hermes provider tools.
The [v5 cohort](../validation/2026-09-19/hermes-native-cohort-v5/README.md)
remains zero of three complete lifecycles on `fd83fb2`. Two attempts left older
conversational copies after exact-ID deletion; one announced a deletion
without calling the tool. Earlier v3 and v4 outcomes remain unchanged too.

Use verified runtime `db932befcd4d0aac9220f9f1e73f043ccd00093c`, wheel
`7df060463b35853d97898ca839b500a66e233bef26b8e52a7de1db799681190a`.
Its checkpoint passes 1,383 source tests, 595 isolated installed cases, four
matching skips in each suite, six source/package checks, and seven replay
assertions. The package exposes `mnemos_forget_preview` and
`mnemos_forget_related`, making eight provider tools. This capability has
fixture/service coverage, but no native-model result at protocol preparation.

Use `hermes_lifecycle_run_v6.py`, `hermes_cohort_run_v4.py`, the unchanged
`hermes_native_bootstrap_v4.py`, and the new `hermes_trace_v2.py`. Commit and
hash these inputs before inference. Preserve and hash the original evaluator
and v3-v5 protocols. The v5 protocol hash is
`56d2fe0c6eb4e3e9013a4f0b341aa9cd33f06de51fe5298e0a13a306b42fd952`.
Neither this run nor its new evaluator may rewrite an earlier outcome.

## Unchanged setup and lifecycle

Retain the v3-v5 model profile, content, scope controls, capture behavior,
restart sequence, replay probe, turn/time budgets, and failure cleanup.
Use local Ollama at `http://127.0.0.1:11434`, as authorized after the remote
machine failed. Check actual model digests and server sampling parameters
before execution and each phase. Qwen3.5:9b uses the existing greedy-v1 alias,
temperature zero, seed 42, presence penalty zero, context 65,536, and no
reasoning. Query embeddings use the pinned Nomic model. Mnemos cognition
modules remain disabled for this host-lifecycle baseline.

The actual pinned Hermes checkout retains only the isolated cwd propagation
patch. Live host source, configuration, credentials, and memory are unchanged.
Optional title-model upgrades remain disabled only in the disposable home.
No controller action may replace a model tool call or repair its arguments.
Supplied descriptions must match all eight schemas from the verified wheel.

Teach, first recall, correction, corrected recall, and final recall keep their
original prompts. Exact content, stable target ID, version progression,
native tool/result correlation, committed capture, retained-or-suppressed
events, scope isolation, and unchanged controls remain required. Writes still
require the supplied asynchronous-job guidance and the original job-status
criterion. A final answer describing an action is not evidence of that action.

## Prospective forget contract

The forget prompt now names the reviewed-copy workflow instead of directing
the model to delete only individually retrieved IDs. The model must retrieve
and inspect the asserted target, request a preview at its inspected version,
review exact/historical and advisory candidates plus source dependencies,
inspect each selected additional fact, and apply the unchanged digest and
matching limit with explicit IDs. Preserve unrelated facts and add no memory.
The controller supplies neither target IDs nor selected IDs. Empty selections
retain their real target-only meaning; they do not imply erasure of all copies.

The new evaluator requires the preview result to precede the model's apply
call. Preview and apply in the same model response do not establish observed
review. It checks schema/status, complete preview, target ID/version, digest,
limit, inspected versions/content for selected copies, explicit required
sources, permitted source selection, and exact agreement between requested
and returned deleted-ID sets. A direct forget call alone cannot pass this
new reviewed-workflow gate. Other lifecycle phases reject the two new tools.

The final state must show every selected fact and source absent, their
evidence and revisions removed, and their suppression records present.
Additionally, after forgetting and after final recall, scan retained scoped
facts, captured events, evidence, revisions, and job records for either label
token: `cedar-ridge-731` and `maple-ridge-842`. Both must be absent from this
project/profile. This strengthens the previous retrieval-only observation;
a top-five result cannot hide a surviving copy. Keep the original final
request check for both labels as well. Foreign and unrelated controls must
remain unchanged. This bounded fixture scan is not universal paraphrase erasure.
Retain the IDs of facts observed in this project/profile across phases so
deletion cannot hide an orphaned evidence or revision row from the scoped
scan. Persist that attribution list in the report. Marker boundaries use
ASCII identifier characters (letters, digits, underscore, and hyphen),
case-insensitively; a different identifier with a shared prefix is not the
same stored label. The older full-request substring check remains unchanged.

A complete preview means its returned selection and source dependencies fit
the bound. Similarity search can remain truncated or nonexhaustive. The native
model must not treat complete as proof that every semantic copy was found.
Incomplete or stale previews cannot be applied; expose failures rather than
loosening the service or evaluator checks.

## Fixed schedule and reporting

Run exactly three independent fresh attempts. Keep the existing eight-turn,
180-second host budget, 240-second outer host deadline, five-second kill
grace, and 1,800-second total attempt deadline. Stop dependent phases after
failure. Launch the next scheduled attempt only after established cleanup.
Do not selectively retry a phase, add an attempt, or edit inputs while the
cohort runs. Record every outcome, including phases not reached.

A pass proves only that bounded attempt. Report all three outcomes together.
This is a package and workflow change, not a controlled causal estimate of
the tools' effect. Module-quality experiments retain their original data and
negative results. Three-host continuity remains a separate test after a bound
Hermes prerequisite passes; Desktop, complete capture, and the pilot remain
separate acceptance gates. Refresh the writer packet with terminal results.
