# Applied encoding comparison: incomplete terminal results

All three arms finished execution on September 19, 2026, using frozen source
`fc72cc251aeafa161a1a390269b8dfa34d83cf2b`, Qwen3.5:9b, Nomic embeddings,
and the unchanged v3 dataset and encoding protocol. Every arm remains
`incomplete`. No label, threshold, missing row, or acceptance gate was changed.

| Arm | Complete observations | Partial observations | Corrections committed / attempted | Forget operations committed / attempted |
| --- | ---: | ---: | ---: | ---: |
| Full | 589 | 11 | 49 / 60 | 60 / 60 |
| No surprisal gate | 592 | 8 | 52 / 60 | 60 / 60 |
| Length/noise replacement | 585 | 15 | 45 / 60 | 60 / 60 |

Each arm records 600 observations across 200 sessions and three repetitions,
covering 2,220 source events and 1,320 automatic extraction jobs. All 1,320
automatic jobs have a completed policy decision in every arm. The remaining
900 source events are outside that policy denominator. Each partial row
contains one `missing_accepted_target` correction error. There are no recorded
provider-timeout or invalid-target-mapping errors. A completed extraction job
can still return no durable fact, leaving a later correction without a target.

The full surprisal gate kept 1,200 automatic events and filtered 120. The
ungated control kept all 1,320. Length/noise kept 1,224 and filtered 96; its
coverage record has 1,320 expected and observed decisions, with zero missing
or unexpected. The two gates disagree on 216 events: length/noise keeps all
120 that surprisal filters, and filters 96 that surprisal keeps. These are
observed filtering decisions, not semantic-quality judgments.

`full.comparator.paired_comparisons` records treatment minus full, with
session-clustered 95% bootstrap intervals. Among the 282 applicable complete
answerable pairs, length/noise's hit-rate difference is -12.41 percentage
points, interval [-19.35, -6.38]. Eighteen incomplete pairs and 300 pairs
without an answerable observation are excluded from that estimate. The
ungated comparison has 285 eligible pairs: -0.35 points, interval
[-1.07, 0.00], with 15 incomplete pairs excluded. These conditional estimates
do not establish an overall result with missing outcomes filled in. Their
exact-text fixture judgments also do not establish general semantic accuracy.

All original gates remain intact, including `real_model: NOT VERIFIED`,
`real_embedding: NOT VERIFIED`, and `human_acceptance: NOT RUN`. The reports
record the actual provider digests, but incomplete execution does not satisfy
their verification gates. Missing token accounting remains unmeasured. No
billing or token-saving claim follows. Concurrent CPU development qualifies
timing results; this was not an isolated latency benchmark.

The six original report/raw-output files are preserved byte-for-byte in
`encoding-reports.zip`. `zip-members.json` binds each member. `manifest.json`
is the terminal encoding manifest; report-embedded manifest snapshots can
still say `running` because they were captured before finalization.
`queue-state-at-archive.json` is a dated observation of the separate retrieval
stage still running, not a terminal queue result. `summary.json` derives
counts without changing raw reports. `evidence-index.json` binds those
artifacts and the exact archive helper. No checkpoint databases are included.
