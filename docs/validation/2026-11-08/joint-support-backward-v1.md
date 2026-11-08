# Adapter and retained-weight backward check

Source `a80e7f90` passes the one-step synthetic retained-checkpoint probe.
The rank-8, alpha-16 q/v adapters cover all eight retained layers; original
parameters stay frozen. The shared head and adapters have 328705 trainable
float32 parameters. Under CUDA autocast their linear computation uses float16;
float32 storage is not a claim of float32 inference math.

Eleven tests pass, including zero-B initialization, gradient flow, separate
float32 merging, nonfinite merge rejection, trainable-parameter enforcement,
CUDA scaled backward/AdamW, early import receipts, timeout evidence and restart
refusal. The review identified four probe gaps, all corrected before execution:
import handling, stage/memory receipts, outer timeout and nonzero learning
gradients. The reviewer confirmed the fixes without executing tests or weights.

One supervised worker completed in 10.41 seconds under the frozen 180-second
outer limit. The synthetic request had exactly 552 tokens and five targets,
using the retained tokenizer and original SDPA/float16 configuration. No corpus
input or label was read. No updated weights or optimizer state were saved.

| Observation | Result |
| --- | ---: |
| Optimizer updates | 1 |
| Synthetic BCE loss | 1.00954056 |
| Unclipped gradient norm | 41.3506813 |
| Adapter-B gradient L1 sum | 830.436728 |
| Classifier gradient L1 sum | 646.451965 |
| Peak CUDA allocated bytes | 1085336064 |
| Peak CUDA reserved bytes | 1138753536 |
| Final gradient scale | 128 |

Every trainable gradient and updated parameter was finite. Clipping used the
declared norm ceiling of 1.0. Every optimizer parameter recorded exactly one
step; original parameter hashes remained equal, trainable hashes changed, and
all frozen file bindings remained unchanged. This establishes this synthetic
backward path and memory footprint, not convergence, full-corpus gradient
stability, semantic quality or complete retrieval latency. The 552-token probe
does not reduce the 4096-token deployment bound.

Private artifacts are under `.venv/joint-support-backward-v1`; freeze SHA256 is
`052db0f5dad56aee0c76c32039300ee8ce134ac92fb15752b7d49695a3e6a1da`.
Result SHA256 is
`e8134f65b0ce4c1908149aaa2f56bf8443ad9bb9fb5eb74bc1788b525ee90bb6`.
The readback-verified `.venv/joint-support-backward-evidence-v1.zip` contains
14 files plus a manifest, SHA256
`1ed57ed198444200739918bd1961709faff9545d93faa851d24e92b5e90f7b88`.

Next: implement the bounded training/evaluation runner and bind corpus-v3,
training weights, exact scaler, unchanged-reranker baseline policy, fixed final
checkpoint and threshold before training. Freeze full-model float16 merged
versus unmerged numerical and exact decision-agreement requirements before
artifact selection; float32 linear tests do not satisfy that requirement.
