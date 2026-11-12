# One local support learning run

This execution implements the unchanged
[numerical recipe](2026-11-12-local-support-learning-recipe-v1.md) after the
[backward gate](../validation/2026-11-12/local-support-backward-probe-v1.md).
The exact source, inputs and runtime must be frozen and reviewed before launch.
One successful computation check does not establish semantic quality.

Use `benchmarks/local_memory/local_support_v2_training.py` with its prepare,
owned and child run modes. All artifacts stay in this checkout's private
environment. The controller requires the frozen clean source revision and
refuses an existing attempt marker or execution receipt. The worker verifies
all file hashes and the unchanged 25-package model environment. No hosted
provider, private API key, normal host profile or live memory store is used.
Local operation stays required; Jev remains a separate explicit option.

Preparation verifies the archived successful backward check, unchanged
numerical recipe, reviewed corpus receipts and original boundary inputs.
It copies the exact target weights, epoch orders and 123 parity requests,
frames every request with the pinned native tokenizer, and freezes the
84-update schedule. Recheck all eleven boundary admissions without inference.
Only 56 training cases and 138 targets can enter gradient updates. Keep the
31 tuning and 29 development cases out of training; their labels and all
lineage, review and critical metadata stay outside inference frames.

The execution uses the full 22-layer float32 master with float16 autocast,
non-reentrant checkpointing, original AdamW groups, fixed loss mass, scaler
128, clipping norm one, seed and learning rates. Compare the unchanged model
and deterministic initial head on all 116 known cases first. Preserve and
restore Torch/CUDA RNG state around that evaluation. The initial comparison
cannot select a checkpoint or threshold. Run all six frozen epoch orders,
four cases per update, exactly 336 microbatches and 84 optimizer updates.
Every step must have finite gradients, positive embedding/layer/head activity
before and after clipping, unchanged scaler and finite parameters/Adam states
at the expected step. Require 336 backbone entries and 672 entries in every
layer, proving forward and checkpoint recomputation inventory.

Record each active case and forward/backward/clip/update phase, completed
microbatch and completed update. Retain append-only microbatch/update ledgers.
A timeout terminal receipt reconstructs complete ledger records newer than
the last progress snapshot, preserving a partial trailing record separately.
An inference failure remains unavailable, retains all unattempted input IDs,
and ends the execution. It cannot earn abstention credit or remove an answer
from its denominator. No OOM, nonfinite or timeout rescue is permitted.

Retain the final checkpoint only. Evaluate its float32 master with float16
autocast on all 123 frames: 116 corpus cases and seven admitted mechanical
boundary controls. Save the final float32 weights and a separate float16
backbone/head export. Explicitly persist float16 in the exported config;
fresh reload with dtype auto must yield finite float16 parameters without a
forced cast masking metadata. Do not mutate the master's parameter precision.
Preserve tokenizer files and the exact recipe with the inactive artifact.

Evaluate that fresh artifact on the same 123 frames with eval mode,
checkpointing off and TF32 off. All IDs, target orders and frame hashes must
match. Require finite logits, maximum absolute error <=0.02 and zero threshold
or selection flips at logit >=0. No adapter merge, threshold change or parity
subset is allowed. Preserve parity failures even if a semantic score looks good.

Reuse the existing diagnostic grader on train, tuning and development for
both initial and final models. Report precision, supported retention, critical
retention/opportunities/distinct facts, answerable recall, no-answer abstention,
exact cohorts, permutation consistency and every false selection by reason.
These are known development results with provisional family ancestry, not
fresh acceptance. Separately retain all eleven repeated boundary controls:
the four unavailable answerable controls and their 17 supported opportunities
stay in the 38-opportunity denominator. Do not count these repeated controls
as independent semantic examples.

Require 10.5 GiB free at precheck and retain the 10 GiB PyTorch allocator cap.
Create the exclusive attempt marker after precheck and before any model load.
The 600-second owned job covers loading, initial comparison, training, final
comparison, export and parity; cleanup must finish within 610 seconds with
zero owned survivors. This cap does not bound CUDA context or total-process
VRAM. The original Windows helper accepts at most 300 seconds, so preparation
produces and hash-binds an exact private copy changing only its timeout guard
to 600. The original helper and all prior source bindings stay unchanged.
No other processes are stopped and no GPU settings are altered.

A completed run is distinct from a semantic pass. Preserve every error,
failed receipt, source/input hash and inactive artifact. No automatic rerun,
model/seed/dataset sweep or semantic rescue follows a failed result. A passing
known development result still needs trained complete retrieval timing,
independent fresh acceptance and the original capture, native, two-week pilot
and delivery obligations. No live activation, package publication or PR merge
is part of this run.
