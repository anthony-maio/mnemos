# Eight-layer learning recipe proposal

Coverage update: [corpus-v3](../validation/2026-11-08/joint-support-contrasts-v1.md)
adds the reviewed constructions required below. Review clears that specific
coverage blocker for a bounded diagnostic attempt. Original training inputs,
labels and weights are unchanged. Adapter/backward tests and exact execution,
scaler and baseline bindings still precede training. The following v2 coverage
assessment records why the correction was necessary.

Status: preparation only. Capacity passed; data coverage and implementation review
still control whether training can start. This proposal does not waive the
existing restriction on sparse or ambiguous development categories. Record any
corpus correction in a new artifact and preserve corpus-v2 and all old scores.

Coverage review confirms that corpus-v2 does not clear that gate. The only
unresolved-condition development contrast is pass-rule/unknown-signature. Both
attribution negatives concern the same gate-approval claim in one retraction
family. Five critical development opportunities collapse to two facts:
pass-rule (four) and build-no-both (one). Before training, add and independently
review two discriminative constructions with supported counterparts: a
quorum-with-veto family where an unknown prerequisite can matter or be overridden
by a known veto, and an attribution family distinguishing denying a report of X
from denying X. These are coverage corrections, not a row quota or fresh
acceptance. Register a new action constraint only if its actual source and
behavior justify critical status; do not promote ordinary facts to inflate counts.

## Objective and data

Use the fixed eight-layer backbone, shared target head and full serializer from
the capacity experiment. Model requests contain no labels, reasons, lineage or
split metadata. Only training labels contribute gradients or weights. All
sources, markers and target IDs remain intact; overflow is unavailable.

The implemented `joint_support_loss.training_weights` first validates group and
label bindings. It gives each connected training family equal total mass, then
divides that mass equally between present binary classes. Negative mass is
divided equally among negative reasons present in that family, then among
queries containing each reason, then among targets of that reason in a query.
Supported mass is likewise divided among supported queries and their targets.
This is a declared macro-weighted objective, not the empirical target frequency
distribution or a calibrated probability estimator. Empty families are rejected.

On corpus-v2 this includes all 282 targets in all 67 training queries. Supported
targets receive 0.5 total mass. Wrong-subject/relation negatives receive 0.152778,
incomplete-source 0.111111, attribution 0.097222, conflict 0.055556, condition
0.041667 and temporal 0.041667. These proportions reflect family coverage;
categories are not globally equiprobable. Individual weights range from
0.000347222 to 0.055555556. Sparse examples can carry high influence, so weights
do not replace label review or create independent evidence.

Loss is the sum of these weights times binary cross-entropy with logits,
computed in float32. No additional positive weighting, oversampling, critical
weight bonus or per-batch normalization is applied. One optimizer update
accumulates gradients over the entire training partition, one query at a time.
Every query appears once per update in a seeded shuffled order. This avoids
randomly excluding rare reasons and makes the normalization explicit.

## Proposed fixed adaptation

Use rank-8 LoRA on q_proj and v_proj in all eight retained blocks, alpha 16 and
dropout zero, plus the shared head. Freeze all original parameters, including
embeddings and final normalization. Initialize LoRA A with the standard linear
Kaiming uniform initialization and B with zeros; initialize the head once.
Use seed 2026110808. Implement and test the adapter separately before activation;
no new dependency or training environment has been installed for this proposal.

Use AdamW with adapter learning rate 0.0001, head learning rate 0.001, betas
(0.9, 0.999), epsilon 1e-8 and weight decay 0.01. No schedule. Clip the unscaled
global trainable gradient norm at 1.0. Use CUDA float16 autocast, float32
trainable parameters and loss, and gradient scaling. Freeze the exact scaler
configuration with the runner. A nonfinite loss, gradient or skipped update
ends the run with a recorded failure; do not silently count it as a completed
step. Fifty optimizer updates are planned, with a hard 30-minute wall-clock
limit. The only selected checkpoint is the final completed update. Timeout,
OOM or interruption is a failed attempt, never an earlier selected checkpoint.

Training memory is separate from inference capacity: the frozen backbone,
trainable adapters/head, gradients, Adam moments, full per-query activations and
CUDA workspace must fit the 16 GB device. Record peak allocation/reservation.
Training requests currently span 105 to 552 tokens; this is not a lower deployed
context limit. A synthetic backward/optimizer wiring check must precede actual
training. Do not change attention, precision, depth or input bounds to recover
from a training failure without a new reviewed decision.

## Selection and evaluation

Freeze the decision rule at finite logit >= 0 (sigmoid >= 0.5), subject to the
existing deterministic authorization/source/version/conflict eligibility gates.
Because loss is reweighted, this threshold is a declared classification rule,
not a calibrated confidence claim. There is no tuning-derived threshold search.
Inspect tuning once after the fixed final checkpoint; retain results regardless
of failure. Development is a separate grouped diagnostic with no training or
checkpoint selection role. Neither partition is fresh acceptance.

Preserve initial untrained prefix/head output as a control and compare against
the unchanged retained reranker using its already frozen scoring policy on the
same development inputs. Do not use random-head behavior as the sole baseline
or transfer old scores onto changed inputs. The exact baseline execution and
eligibility contract must be bound in the runner before training.

Report per-target precision, supported/critical retention, answerable recall,
no-answer abstention, whole-cohort accuracy, permutations, category mistakes and
unavailable cases. Sparse critical facts are named separately; a perfect score
on two development facts cannot establish the original critical-retention gate.
Retain failed and all-abstention outcomes. Original quality thresholds, complete
retrieval-plus-support timing at 10000 accepted facts, independent fresh
acceptance minima and repeated runs remain unchanged. Merge adapters for the
candidate deployment artifact only after testing numerical output/decision
agreement; the merged artifact needs its own complete-path timing.
