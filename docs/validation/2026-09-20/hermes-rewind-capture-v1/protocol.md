# Hermes rewind capture boundary v1

Run the identical API fixture against the isolated cwd-only host and the
existing durable-identity candidate, both based on Hermes `14a3463`. Use the
verified Mnemos `4bbab4f` wheel, SHA256
`9af6667d86933570beecc940e55e637f70a9258e5339e948dc48ba9692782bf6`.

The runner is `benchmarks/native_hosts/hermes_rewind_capture_v1.py`.
Each arm gets a fresh output directory, Hermes home, SessionDB, and outbox.
Network connections are prohibited. No live host files, configuration, or
memory data may change. Bind the runner, host source files, wheel, raw logs,
and report; retain both arms regardless of outcome.

For both actual CLI `undo_last` and `retry_last` methods:

1. Persist two turns in actual SessionDB and capture their native row IDs
   through actual MemoryManager and the installed-wheel Mnemos provider.
2. Replay the transcript and require four byte-equivalent queued payloads.
3. Strip IDs from the warm CLI view, then invoke the actual rewind method.
4. Persist an identical replacement turn with SessionDB. Its new IDs must
   differ from the removed suffix. Pass the CLI's retained warm prefix and
   the newly persisted suffix through MemoryManager.
5. Require exactly six queued observations, unchanged original payloads,
   native identities on the retained prefix, and no provisional observations.
6. Restart MemoryManager/provider, reload durable rows, and replay a shortened
   retained view. Require identical payloads and no additional observations.

The expected unpatched failure is lost retained IDs, causing mixed-identity
provisional captures. The candidate must pass both cases with unchanged host
source and zero attempted network connections. If either expectation fails,
retain that result and investigate it rather than altering acceptance.

This is host API and queued-outbox evidence. Turn generation and persistence
are fixture-driven, not a native model conversation. It does not prove HTTP
delivery, service commitment, extraction, process-crash recovery, compaction
clone provenance, or migration of previously captured ID-less observations.
An initial native-row capture is an explicit precondition.
