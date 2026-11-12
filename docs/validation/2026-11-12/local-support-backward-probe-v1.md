# Full-encoder backward check

The repaired synthetic check passed on source
`34748fd340d91bf77fbba0e82424979a51c81602`. It completed four full
8,192-token microbatches and one disposable optimizer update using all 22
layers and 149,014,272 backbone parameters. This permits implementing and
reviewing the single corpus runner. No corpus training, semantic grading,
trained latency measurement or live installation occurred.

The first execution at `dcaf3fac` failed during tokenizer import because its
sanitized Windows environment omitted `USERNAME`. It loaded no model weights
and completed no forwards or updates. Its failed terminal receipts and source
are preserved in a verified private archive, SHA-256
`5d72708cf8535288fd9fa0960685cdc2157564c7c8108b0522b0f8aaad775b83`.
Zero PyTorch allocated/reserved bytes does not establish zero CUDA context
allocation.

The sole allowed code-only repair preserves that Windows identity variable
and sets a private Torch import-cache directory before child imports. All
five prepared artifacts are byte-identical between attempts: training weights,
epoch orders, probe request, complete frame and deployment parity requests.
Recipe, tokenizer binding and 25 dependency versions are unchanged. The repair
does not alter precision, model, data, input length, loss or resource limits.
Sixteen focused runtime contracts passed before the repaired execution.

The repaired child completed in 39.02 seconds; the enclosing job took 40.15
seconds and the controller 41.29 seconds. All five processes in the owned job
were stopped after cleanup. The 78 file bindings, source revision and runtime
inventory remained unchanged. The terminal `result.json` and controller are
authoritative; intermediate progress snapshots retain a pessimistic failure
status until terminal success is written.

All four inputs retained five targets and eight complete sources per target.
The model recorded four backbone entries and eight entries in each encoder
layer, including checkpoint recomputation. Every trainable gradient existed
and was finite; the embedding, every layer and classifier had positive
activity before and after clipping. The unclipped norm was 7.8076849. Updated
parameters and Adam states were finite, every optimizer state reached step
one, the scale remained 128, and both backbone and classifier changed.

Peak PyTorch allocation was 2,812,061,696 bytes and reservation was
3,047,161,856 bytes, below the unchanged 10 GiB allocator cap. This is allocator
accounting, not a total-process GPU memory limit. No disposable weights were
saved. The verified repaired archive has 68 members and 360,983 bytes, SHA-256
`ac60b110ad32c71aaca095128d1f64e087b924e68aca90b4c362422dc830b012`.
Large model/runtime files are hash-bound rather than duplicated in the archive.

The next step is one frozen corpus runner with the reviewed 84-update recipe,
initial/final comparison and exact float16 export parity. Local operation stays
required and Jev stays explicitly optional. Original quality, complete trained
timing, fresh acceptance, capture, native, pilot and delivery obligations remain
open. This computation check does not qualify any of them.
