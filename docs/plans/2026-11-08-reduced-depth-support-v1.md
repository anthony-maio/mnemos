# Fixed eight-layer joint support candidate

This is a new supervised model hypothesis. Retain exactly blocks 0 through 7
of the bound 28-layer Qwen3 backbone, its embeddings, rotary encoding and final
normalization, followed by the existing shared target head. No claim of
equivalence to the original model is made. Eight layers is fixed before timing:
the rough linear estimate from the failed 876.17 ms full-depth result is 250 ms,
leaving a provisional margin inside the 500 ms complete recall budget. That
estimate excludes fixed costs and is not a measured result or a latency proof.
There will be no depth sweep after failure.

The hypothesis is that supervised adaptation can learn evidence support from
the retained prefix. Losing deeper representations may destroy the required
condition semantics. Capacity alone cannot establish quality. Subject identity,
negation, reported versus adopted constraints, requested hypotheticals, unknown
conditions, conflicts, temporal scope and source ownership all remain required.
Preserve the reviewed corpus and grouped splits. No training or fresh acceptance
is authorized by this document. A passing screen only permits preparation of the
separate frozen learning recipe and its data review.

Use the same six synthetic workloads, full 4096-token/64000-byte serializer
bounds, five targets, up to eight sources each, overflow rejection, float16,
native SDPA, device and dependencies as the original capacity runner. Preserve
every source and trusted target marker. Do not alter the original runner or its
closed evidence. The separate runner freezes its own source, this plan, the
prefix helper, head, serializer, 12 model files, requests and frame bindings.
Verify the full checkpoint loading before discarding blocks. Record original
and retained depth. Use one exclusive-start run, five warmups and fifty measured
repetitions per shape, including serialization, transfers, head and synchronized
GPU completion. Preserve failed samples and interruptions. All six shape p95s
must be below 500 ms for this necessary computation-only screen to pass.

Passing still requires semantic development, independently frozen fresh quality
acceptance, and installed retrieval plus support at 10000 accepted facts within
500 ms. The original 95% precision/critical retention and 90% recall/abstention
gates remain. Capture latency, native authorization acceptance, publication and
the original 14-day pilot remain separate outstanding obligations.
