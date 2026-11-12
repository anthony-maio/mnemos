# One controlled claim-span learning comparison

The [claim-span contract](../validation/2026-11-12/claim-span-readout-v1.md)
passes its model-free gate. This is a new representation decision after the
closed [marker-readout failure](../validation/2026-11-12/local-support-learning-v2.md),
not a rerun or rescue of that candidate. Independent review of the exact
numerical/execution freeze is required before launch.

Test whether reading complete claims provides useful support decisions on
the same known corpus. Preserve every query, date, candidate and complete
owned source. The representation change includes its preamble, removal of
tail markers and FP32 claim averaging. This experiment cannot isolate a causal
effect of averaging alone. Source ownership, decisive qualifiers and order
sensitivity remain semantic risks; successful framing does not resolve them.

## Fixed comparison

Use `claim_span_learning.py` and its committed protocol. Bind the original
numerical recipe by SHA-256
`9660b2f80ff5ba4c98d8dbfa270594bf134ee59b3aee6fa64907a768c3cf2f6a`.
The numerical dictionary remains byte-bound and unchanged: full 22-layer
149,014,272-parameter ModernBERT backbone, hidden size 768, shared scalar head,
original checkpoint/revision, seed 2026111101, deterministic independent head
generator and uniform initialization. Keep FP32 master parameters, FP16
autocast, SDPA, TF32 off and non-reentrant checkpointing. Keep original AdamW
rates/groups/decay, fixed family/class/reason/query/target mass, multiplier 14,
four-case accumulation, norm-one clipping and scaler 128. No parameter freezing,
adapter, truncation, source deletion or new threshold.

Copy the same training weights, epoch orders and full schedule byte-for-byte.
Only the original 56 training cases and 138 targets enter the six epochs,
336 microbatches and 84 updates. Reconstruct every saved input through the
actual pinned tokenizer before use; labels, lineage, split and critical flags
never enter inference frames. The adapter preserves backbone/classifier state
names and accesses one complete-context encoder output per cohort. Pool every
mapped claim token in FP32, in candidate order, and apply the same scalar head.

The original recipe's historical marker-prelude and 123-frame parity metadata
describe its parent experiment. This protocol controls the new inputs and
126-frame parity inventory; it does not silently inherit those old admissions.
Numerical optimization settings remain identical. Export metadata records both
the numerical reference and this execution protocol, plus the claim readout
schema/source hash. Reload rejects inconsistent readout metadata.

## One contained execution

Use one exclusive private run root, frozen clean source and one Windows-owned
job. Require 10.5 GiB free before the attempt marker and retain the 10 GiB
PyTorch allocator cap. It does not bound total-process VRAM. The entire job
has 600 seconds, including loading, prelude, reset, training, comparison,
checkpoint export and reload/parity. Cleanup must complete within 610 seconds
with zero owned survivors. Do not stop unrelated processes. Bind the existing
timeout-only 600-second supervisor copy; leave its original source unchanged.

First perform four complete 8,192-token synthetic forwards/backwards on the
new exact-token frame: five claims with eight sources each. Reuse the original
false/true/false/true/false synthetic labels, target weights 1/280 and one
disposable update. Require four backbone entries and eight per-layer entries,
finite gradients/parameters/Adam states, positive embedding/layer/head gradient
activity before and after clipping, unchanged scaler and changed body/head.
This stage is not a semantic or latency result.

Delete the disposable model, optimizer and scaler, clear unused allocator
cache, reseed and freshly reload the original checkpoint/head. Require both
initial parameter digests to match their values before the synthetic step;
the head digest must be
`da88e30e11194dee428b7f10f1f9fabe04fad4283641d40c0f48afb1415943b2`.
No synthetic update or optimizer state may enter corpus training. This
reset is a gate within the single job, not another candidate attempt.

Evaluate the restored initial model on all 116 known corpus cases. Preserve
and restore Torch/CUDA RNG state around that comparison. Then run all 84 fixed
updates. Every update must satisfy the original finite-gradient/activity,
clipping, scaler and Adam-state checks. Require 336 backbone entries and 672
entries in every layer during training. Record active phases/cases and
append-only completed microbatch/update ledgers. Reuse the bounded Windows
atomic-write retry; monitor append-only ledgers during execution.

Keep the final checkpoint only. Evaluate the final FP32-master/autocast model
and freshly reloaded FP16 artifact on exactly 126 complete frames: 116 known
cases, eight admitted historical controls, the new exact-token boundary and
the non-gold Unicode/repetition fixture. Both use eval mode, checkpointing
disabled and TF32 off. Require all identities, orders, frame hashes and finite
logits, maximum absolute error <=0.02 and zero threshold/selection flips.
The scalar threshold remains logit >=0. Save final master and inactive export,
including consistent float16 config, recipe and readout metadata.

## Outcomes and stopping rules

Independently reconstruct the full schedule, prelude/reset evidence, all
partition grades, raw selections, permutation results, boundary accounting,
artifact hashes and cleanup. Compare the previous marker result and the new
initial/final span results without treating the known corpus as fresh data.
Keep all eleven historical boundary requests in their denominator, using
their new measured admissions only for this representation. No derivative
boundary or Unicode fixture adds independent semantic evidence.

Original quality gates remain precision and critical retention >=95%,
answerable recall and no-answer abstention >=90%; the diagnostic also requires
supported retention >=95%, complete availability and consistent permutation
groups. Retain every failure even if another metric improves. No threshold,
seed, checkpoint, epoch or data sweep follows a failed result. An infrastructure
failure is preserved and needs a reviewed correction before any restart;
OOM, nonfinite, timeout or semantic failure does not authorize a rescue.

Computational completion, artifact parity and semantic quality are separate
outcomes. A diagnostic pass still requires trained complete retrieval timing
at 10,000 facts, the original fresh acceptance and capture/native/two-week
pilot/delivery obligations. No publication, PR merge, pilot refresh or live
configuration change occurs here. Local operation stays required; no key or
hosted provider is used and Jev remains explicitly optional.
