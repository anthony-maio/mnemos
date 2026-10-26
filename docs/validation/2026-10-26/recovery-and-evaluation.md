# Recovered work and the current evaluation result

The September 23 judged run already fixed the correction failure described in the September 22 handoff. It completed 600 of 600 observations, including 60 corrections and 60 forget operations. Its retrieval evidence still does not justify recommending automatic extraction or the full biomimetic profile by default.

This review made no provider calls, reran no model evaluation, changed no thresholds, and edited no frozen results. The accompanying [provenance record](recovery-and-evaluation.json) binds the source archives, recovered files, corpus, execution manifests, and raw result files by SHA256. Raw model outputs and databases are not copied into this report.

## Source recovery

The delivery branch starts from `feat/typed-judgment-layer` at `b44ee7626ccbb32d3ba4358b65ad10b9f353d977`. That branch contains the functional local memory service, source-bound extraction, host adapters, and typed judgment layer. The pre-September `master` checkout and its unpublished commits were preserved.

The later bundle `mnemos-heldout-abce821-v7-judged-typesafe-full` identifies source commit `abce821116bf615d0a0f6a67ded772ae0114109d`. That commit was absent from the available git refs and the recorded LFS rewrite map. Its archive and preparation hashes match the execution binding; the four recovered files match the preparation manifest byte for byte:

- `mnemos/extraction/judged_classification.py`: explains that a durable fact introduced by a request to remember remains a fact; preserves quotation and temporary-instruction boundaries.
- `benchmarks/local_memory/heldout_run.py`: uses HTTPX for the hosted model identity probe.
- `benchmarks/local_memory/colab_heldout.py`: supports full-only summaries and exposes precision counts and correction completion, including failed attempts.
- `benchmarks/local_memory/typesafe_spike.py`: accepts development fixtures without a historical baseline.

No other application files were recovered from the bundle. Original branches, source archives, and evaluation outputs remain intact. Offline regression tests cover the reporting contract and source rendering boundaries. Mocked classification tests do not establish model quality.

## What was measured

The bundle names use `v7`; the corpus manifest declares `local-memory-heldout-v6`. Its content hash is the authoritative identity. It contains 20 development sessions and 200 held-out sessions, all synthetic, with three repetitions in the reported runs. Human acceptance was not measured. The latest judged run reused the same corpus after the original failures had been inspected, so it is a regression rerun, not a fresh blinded evaluation.

| Run | Complete / attempted | Corrections committed | Forgets committed | Exact precision proxy | Critical retention | Exact answerable recall | No-answer accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original judged baseline | 600 / 600 | 60 / 60 | 60 / 60 | 1110 / 3900 | 1110 / 2100 | 231 / 600 | 240 / 660 |
| Original judged full | 540 / 600 | 0 / 60 | 60 / 60 | 990 / 1792 | 990 / 1890 | 201 / 540 | 201 / 600 |
| Local source-bound full | 531 / 600 | 60 / 60 | 0 / 60 | 1033 / 2312 | 1033 / 1841 | 230 / 531 | 135 / 531 |
| Latest judged full | 600 / 600 | 60 / 60 | 60 / 60 | 1110 / 2150 | 1110 / 2100 | 231 / 600 | 150 / 660 |

Quality columns use complete observations. Lifecycle columns include every attempted operation. The original judged full excludes 60 correction sessions. Local source-bound full excludes 69 sessions: 60 missing forget targets and 10 maintenance errors, with one overlapping session. These exclusions must travel with any quoted comparison. The latest judged run is full-only; it does not replace the original seven-profile ablation matrix.

Precision is an exact frozen-text, scope, and evidence proxy. It is not semantic factual precision. Latest judged recall is 38.5% under the unchanged exact comparison; no-answer accuracy is 22.73%. Its corrections and forgets pass this synthetic lifecycle fixture, not a real-host pilot.

## What the raw results explain

The original judged classifier stored no facts for 254 of 300 automatic captures phrased as a request to remember. The later recovered description change corresponds to all 60 correction targets being available in the new run. This repair was already tested in September; repeating the same prompt edit would not address the remaining problem.

A separate, read-only answer-span diagnostic found the entire literal gold sentence somewhere in the returned text for 586 of 600 latest judged answerable retrievals. Only 231 returned texts equal it exactly. The other 355 contain an additional prefix or suffix, often `A related decision detail:`. This does not replace the frozen score, establish semantic correctness, or measure precision. It shows that exact span boundaries account for much of the reported recall deficit. The provenance record retains both counts under separate names.

Abstention has a different problem. With extraction disabled, automatic captures are stored as `captured_passthrough`. The default `memory` retrieval view excludes those records. The harness uses that default view for both baseline and full, while extraction promotes full-profile records into the searchable memory set. These arms therefore differ in which facts are visible. The original comparison combines extraction, visibility, and ranking; it cannot isolate retrieval quality.

There are also concrete irrelevant results. For an unknown incident-commander query, the judged profile returns the transient status `The first check passed, but the deployment note still needs a human review.` The same status appears in 15 of the 22 ordinary no-answer regressions in the original judged full run. Returning related text is not evidence that the requested answer is present.

## A bounded next diagnostic

Keep explicit remember, correct, forget, and restart persistence independently useful. Leave automatic extraction and biomimetic ranking experimental until their added cost produces a measured benefit.

For retrieval, use a separate development fixture and one immutable accepted-fact snapshot for all arms. Preserve scope, source authority, candidate contents, embeddings, top-k, and context budget. Record semantic candidates before affective and graph scoring, then evaluate answer support and abstention separately from ranking. Include absent entities, absent relationships between known entities, transient status, and quoted instructions. Freeze the rubric before running a provider.

The current code checks `score.semantic` against the relevance floor before selection. State and graph boosts cannot admit a candidate below that floor. The lexical path forms a candidate union; it does not currently supply a fusion score. An initial hypothesis that boosts alone bypass the floor is therefore unsupported. A semantically related candidate can still pass the floor without answering the question, and recent-state matching plus graph energy can rank it highly. Test a bounded answer-support check on the fixed development candidates before tuning thresholds or changing the production retrieval views.

The original ablations did not establish useful gains for the other four modules. Preserve that negative result. A future module claim needs a prospective comparison with comparable visible facts and the original failure counts intact.
