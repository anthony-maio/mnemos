# One numerical recipe and disposable backward/memory probe

This is the prospective numerical specification; execution status belongs in
the [current goal record](2026-11-06-mnemos-improvements.md).
The reviewed 116-case corpus permits this
recipe and backward-memory review, not corpus training yet. The old failures,
original quality/fresh-acceptance gates and real daily-use pilot remain intact.
Local operation is required; Jev is explicitly optional and is not used here.

Use [the committed recipe](../../benchmarks/local_memory/local_support_v2_recipe.json)
and the existing exact ModernBERT checkpoint, tokenizer and trusted full-source
frame. Retain all 22 layers and all 149,014,272 backbone parameters. Add only
the shared 768-to-one linear head at the trusted per-target decision positions.
The inherited NLI task heads remain unused. No layer reduction, adapter,
truncation, source removal, compilation or alternate attention backend.

Set global Torch/CUDA seed 2026111101 before checkpoint loading. After loading,
create the head on CPU and overwrite its default initialization using an
independent CPU generator seeded 2026111101: uniform in [-1/sqrt(768),
1/sqrt(768)], drawing weight then bias. Move that float32 head to the device.
This freezes initialization independently of checkpoint-loader RNG consumption.
The Python shuffle RNG is separate. Record the exact initial head digest.

All backbone/head parameters train in float32 with float16 autocast, SDPA and
non-reentrant gradient checkpointing. Disable TF32. Use four intra-op and one
inter-op CPU threads. Keep the explicit one-thread embedding policy for later
complete retrieval measurement; no embedding provider runs in this probe.

AdamW uses backbone learning rate 0.00002 and classifier learning rate 0.0001,
betas 0.9/0.999 and epsilon 1e-8. Weight decay is 0.01 only for tensors with at
least two dimensions; all one-dimensional biases/norms have zero decay. Use
constant rates, foreach=false and fused=false. Every trainable parameter must
appear once in exactly one of four named groups. All parameters are trainable.
Use gradient clipping at norm 1.0 after unscaling. GradScaler starts at 128,
growth factor 2, backoff 0.5 and growth interval 2,000; any skipped/nonfinite
step fails this fixed attempt rather than changing scale policy. Clipping must
use error_if_nonfinite=true and return a finite positive norm; every layer,
embedding and head must still have nonzero gradient activity afterward. Updated
parameters and all Adam state tensors must be finite. Weight-decay-only changes
cannot establish a successful gradient update.

The existing `training_weights` function supplies exact target loss mass:
equal connected-family mass, then present classes, negative reasons, queries
and targets. On the reviewed training partition, 138 weights sum to one and
positive/negative masses each sum to one half. There are eleven groups and
56 queries. No reserved labels or critical flags influence this weighting.
Persist every target weight, group/class totals and source bindings before
execution. This keeps repeated rows from dominating without changing labels.

Visit each of the 56 sorted training IDs once per epoch, shuffling with one
Python RNG seeded at 2026111101 across all six epochs. One cohort per forward,
four cohorts per optimizer update, six epochs and exactly 84 updates. For each
microbatch, sum target BCE-with-logits times its fixed weights and multiply by
56/4 = 14 before scaled backward. At fixed parameters this has the correct
expected full-objective scale under uniform cohort sampling. Random reshuffling
does not make each gradient conditionally unbiased after preceding updates.
Do not average by target count or renormalize each query. All 336 cohort visits
are retained.

Keep the final checkpoint only and select logits >=0, equivalent to sigmoid
>=0.5. No tuning-based threshold or checkpoint selection. Compare final and
unchanged initial backbone/head on identical known partitions; initialization
is a negative control, not a useful pretrained support baseline. Retain all
errors. Float16 deployment must keep every target ID/frame, finite logit,
absolute error <=0.02 and zero threshold/selection flips against the float32
master evaluated with the same autocast. No adapter merge is involved.
Freeze parity on all 116 reviewed requests plus all seven admitted boundary
controls, including the complete token/byte bounds and the empty bypass. Both
sides use eval mode, checkpointing disabled, TF32 off and the fixed seed. Compare
the float32-master/float16-autocast reference with a freshly reloaded float16
artifact. No parity subset or threshold is chosen after seeing outputs.

Before that corpus attempt, permit one disposable optimizer check using the
existing exact-token boundary control: five facts, eight complete sources per
fact and exactly 8,192 native tokens, including all markers. Use the same model,
head, optimizer, precision, checkpointing, clipping and scaler. Accumulate four
identical full-length synthetic microbatches and perform one optimizer step.
Synthetic labels are false/true/false/true/false, with each target weight 1/280.
Together with multiplier 14 this yields a mean-sized objective after four
microbatches. These synthetic labels measure computation, not answer support.

All trainable gradients must exist and be finite. Each of the 22 layers,
embedding and classifier must have nonzero gradient activity; the body and
head must change, all optimizer state steps must be one, and scaler must remain
128. Require four backbone forward entries and eight pre-hook entries per
encoder layer, covering four forwards and four checkpoint recomputations in
the pinned implementation. Record peak CUDA allocation/reservation.
Do not save the disposable weights or use its outputs as semantic evidence.

The shared GPU currently has other allocations. Give this isolated process a
fixed 10 GiB allocator cap, not a hard total-process VRAM limit, and require at
least 10.5 GiB free at launch. A
failed free-memory precheck is not a started candidate attempt; preserve the
receipt and wait for external capacity. Create an exclusive attempt marker
after that precheck and before checkpoint loading or model GPU allocation.
A loading OOM therefore consumes the attempt. After the marker, an OOM,
nonfinite or timeout ends this recipe under the declared resource budget.
Do not rescue it by reducing tokens, sources, depth, precision or microbatches.
Run inside a Windows-owned job, bounded to 180 seconds plus cleanup within
190 seconds, and verify zero owned survivors. Do not stop other processes or
alter host GPU settings. The later corpus run is bounded to 600 seconds plus
cleanup and requires its own exact runner freeze after this gate passes.

Freeze actual code, recipe, full input/frame, model/tokenizer assets, reviewed
data, weight table, interpreter/dependency inventory and owned-job controller.
Use the unchanged 25-package model environment and offline assets. Save progress,
all four observations and failed terminal receipts. Verify bindings and process
cleanup after either outcome. One code-only infrastructure repair may follow a
failure; that allowance excludes OOM, nonfinite and timeout failures. A numerical
or resource-policy change requires a new explicit design,
and no semantic failure authorizes a sweep or rescue.

A passing probe permits the one corpus runner to be implemented, frozen and
reviewed. It establishes no quality or timing. Any later development pass still
needs exact artifact parity, complete trained retrieval timing at 10,000 facts,
independent fresh acceptance and the original capture/native/pilot/delivery
obligations. No live installation, project pilot file, package or PR is changed.
