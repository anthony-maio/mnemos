# Controlled claim-span learning result

This configuration failed both semantic quality and numerical export parity.
All 84 corpus updates completed after the full-length synthetic check and
checkpoint reset. The new readout fit many training cases, but final
development precision is 3/5, recall/retention 3/24 and critical retention
1/8. The candidate is inactive and closed. No threshold, precision, seed,
epoch, checkpoint or data rescue follows this result.

Source is `6bfc82e20fec7eb5c6d7a8e451f8dec3df0a0800`, following the
[prospective protocol](../../plans/2026-11-12-claim-span-learning-v1.md).
The parent numerical recipe is unchanged, and weights, epoch orders, complete
84-update schedule and supervisor are byte-identical to the prior experiment.
Only the representation changes: full-claim pooling, its preamble and removal
of tail markers. This is not an isolated causal test of pooling alone. All
original requests, labels and owned evidence remain intact.

## Known-partition quality

Final master and freshly reloaded float16 artifact make identical selections.

| Partition | Precision | Supported retention | Answerable recall | Critical retention | No-answer abstention |
| --- | --- | --- | --- | --- | --- |
| Train, 56 cases | 35/35 = 100% | 35/40 = 87.50% | 33/38 = 86.84% | 9/9 = 100% | 18/18 = 100% |
| Tuning, 31 cases | 9/11 = 81.82% | 9/27 = 33.33% | 9/26 = 34.62% | 6/12 = 50% | 5/5 = 100% |
| Development, 29 cases | 3/5 = 60% | 3/24 = 12.50% | 3/24 = 12.50% | 1/8 = 12.50% | 5/5 = 100% |

The span model's initial development scores were 0/2 precision, 0/24 recall
and 0/8 critical retention, with 5/5 no-answer abstention. Its initial training
precision/retention were 3/11 and 3/40. The final training fit is substantially
stronger, but reserved performance remains poor. Compared with the previous
marker artifact, development precision rises from 9/29 to 3/5 while retained
supported facts fall from 9/24 to 3/24. This is not an overall quality pass.

The original precision/critical-retention gates remain >=95%, recall and
abstention >=90%, with diagnostic supported retention >=95%, complete
availability and consistent permutation groups. All 116 known corpus cases
were assessed. Critical denominators are supported opportunities: the three
partitions contain three, seven and four distinct critical facts. The five
no-answer development cases do not establish fresh acceptance or uncertainty
at the original sample sizes.

The universal last-candidate pattern is absent. Only five development cohorts
receive a selection, occupying positions one through five once each. All six
development permutation groups are consistent, but abstaining across variants
can also produce consistency. One of six tuning groups remains inconsistent:
`linnet-trace-spans-b-duration` / `linnet-trace-spans-permutation`. Development
false support includes one wrong-subject/relation and one incomplete-source
selection; tuning has two attribution mismatches. Every missed fact, critical
opportunity and false selection is retained in the
[receipt](claim-span-learning-v1-receipt.json), alongside
[raw predictions](claim-span-learning-v1-predictions.json).

Boundary retention is 0/38 supported opportunities. All eleven repeated
historical controls remain counted, including 12 opportunities in three
currently unavailable requests. They are not independent semantic examples.
The historical token-overflow request is admitted only under the new format's
measured token count; the earlier format's rejection is unchanged.

## Computation, reset and export

The synthetic prelude completed four complete 8,192-token backward passes and one
disposable update, with five claims and eight sources each. It recorded four
backbone entries and eight entries per layer, positive embedding/layer/head
gradient activity before and after clipping, finite state and scaler 128.
The model/optimizer/scaler were discarded, RNG reseeded, and original weights
reloaded. The bound runner checked both initial parameter digests before
corpus training. Reset success is recorded, but the individual before/after
backbone digests were not persisted; that part of the evidence rests on the
reviewed runtime assertion. The deterministic initial head digest matches
`da88e30e11194dee428b7f10f1f9fabe04fad4283641d40c0f48afb1415943b2`.

All 336 corpus microbatches and 84 updates pass the fixed gradient, clipping,
scaler and finite-parameter/Adam-state checks. Training records 336 backbone
entries and 672 entries in each of 22 layers. The backbone and head change.
Per-update state finiteness is enforced by the frozen assertions; disposable
optimizer states are not preserved as separate weight artifacts.

Numerical export parity fails: 25 target logits across 18/126 complete frames
exceed absolute error 0.02, with maximum error 0.0986328125. There are zero
sign flips across all 387 target outputs and no changed selections. Identical
decisions do not waive the prospectively frozen numerical tolerance. Final
FP32 master and inactive FP16 deployment weights remain private and hash-bound;
metadata records the actual claim readout and the numerical/execution policy.
No export repair or alternate deployment precision was attempted.

The child failed at the parity gate after 99.62 seconds; the job took 100.84
seconds and the controller 102.09 seconds. Peak PyTorch allocated/reserved
memory was 2,812,061,184 / 3,047,161,856 bytes, below the 10 GiB allocator cap.
This is not total-process VRAM accounting. The controller verifies all 135
file bindings, 25 runtime versions and unchanged source, with zero owned
process survivors. No trained complete retrieval latency was measured.

Five focused CPU contracts passed before launch, including native-frame
reconstruction, empty bypass, parameter namespaces and a small-model export/
reload test. Independent post-run review reconstructed the prelude, corpus
schedule, all nine partition grades, parity, artifact hashes/headers and
cleanup without discrepancies. The readback-verified private evidence archive
has 102 members and 3,836,405 bytes, SHA-256
`8bf758cf719f77be31b44cc2d52dfc97a619bbc755ad3890418798ddbc3ff5f5`.
Large final weights remain separately retained with verified hashes.

## Goal state

The known corpus does not qualify fresh acceptance. Neither the current
artifact nor the previous marker model qualifies for activation or trained
timing work. No automatic minor-head, model, threshold or data sweep follows.
Preserve both failures and the mechanical contract; any future semantic
experiment needs a substantive prospective decision.

The local memory workflow and explicit Jev option remain unchanged. No hosted
call, live database/configuration change, pilot refresh, package publication
or PR merge occurred. There is still no combined quality/timing pass and zero
qualifying original-pilot days. The full six-priority goal remains active,
with capture, fresh/native acceptance, the actual two-week pilot and delivery
requirements outstanding.
