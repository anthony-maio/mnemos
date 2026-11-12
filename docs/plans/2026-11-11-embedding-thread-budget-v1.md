# One local embedding resource comparison

Status: [the sole comparison failed recall non-regression](../validation/2026-11-11/embedding-thread-budget-v1.md).
All vectors and 50 result pairs matched, and idle CPU fell to zero. Candidate
p95 rose from 150.04 to 161.43 ms. Automatic allocation remains the source
default; the thread budget is retained as opt-in only. The prospective protocol
below is historical and does not authorize a retry or default change.

The current installed embedding session uses automatic ONNX thread allocation.
A single cached BGE embedding took 6.1142 ms, followed by 5,796.875 ms of process
CPU time during the next approximately one second of wall time with no requests.
Before session creation and after its workers settled, the process used no
measured CPU while sleeping. This is one resource diagnostic, not evidence that
it caused any earlier complete-path tail or that a latency gate now passes.

Use FastEmbed's existing public threads argument to default local interactive
embedding to one thread. Expose `embedding.threads` and
`MNEMOS_EMBEDDING_THREADS`: one is the default, positive integers set a budget,
and zero explicitly selects upstream automatic allocation for batch workloads.
Reject booleans, negatives, null and nonintegers before model initialization.
Do not change model weights, tokenizer, role prefixes, normalization, dimension,
embedding identity, candidate visibility, scope, evidence, durability or Jev.
This changes the source default only; existing installations and pilots stay
unchanged pending their own qualification and activation.

Before adoption, run one frozen baseline/candidate pair in fresh owned processes.
Baseline is the installed `9c326cbc` wheel; candidate is committed source with
the thread-budget change. Both use the same external 57-package environment,
cached BGE assets and private copies of the same 10,000-accepted-fact fixture.
No support classifier, CUDA model, hosted provider or live memory is involved.
The closed resident support configuration is not rerun or regraded.

Freeze the helper, source, package manifest, dependency file bindings, embedding
assets, requests and protocol before execution. Use one 180-second Windows-owned
job and fresh outputs. No retries or parameter sweep follows failure.

In each arm, record idle process CPU before loading, one unrelated warmup, one
batch of eight fixed synthetic texts, and each of those texts individually.
The texts cover empty, ordinary facts, negation, conditions, attribution,
Unicode escapes and a long repeated input. Save the complete finite vectors.
Require matching dimensions, maximum absolute error <=1e-6 and cosine >=0.999999
between corresponding vectors. Report whether outputs are also bit identical;
do not relabel tolerance agreement as bit identity. Require the candidate's
first 500 ms of idle process CPU after the final embedding to be <=50 ms and
at least 90% below baseline. CPU intervals sum across process threads and are
not wall-clock durations. This is a bounded diagnostic, not a power guarantee.

Then use the unchanged complete client.retrieve boundary for 50 measured ordinary
recalls after three warmups per arm. The fixture log contains 25 measured recall
requests; replay two consecutive passes over those same 25 requests, in the
same order in both arms. This is reused diagnostic data. Include actual
BGE, scoped retrieval, access writes, source assembly, transport and decoding.
Record all results and failures; stop on an operational failure and retain
unattempted denominators. Require identical returned fact IDs/content/versions,
sources and context for every pair. Ranking scores may differ numerically and
must not be used to excuse changed selection. Use inclusive interpolated p95,
require candidate p95 <=baseline p95, and report both against 500 ms without
calling this a full support-enabled latency or semantic-quality pass.

Run model-free config/runtime/embedding checks before freezing. Verify source,
installed files, dependencies, original fixture, assets, SQLite integrity and
owned process cleanup afterward. A resource or numerical-parity failure stops
default adoption for this increment. It does not authorize relaxing original
quality, timing, fresh-acceptance, native, pilot or publication requirements.

Primary references checked November 11:
[ONNX threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
documents physical-core defaults and enabled spinning;
[FastEmbed 0.4.0's public constructor](https://github.com/qdrant/fastembed/blob/v0.4.0/fastembed/text/text_embedding.py)
already accepts and forwards threads. No dependency change is needed.
