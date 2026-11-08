# Replay acknowledgement comparison

The candidate has lower observed tail latency in this synthetic comparison,
but no typical-latency improvement and no capture-gate pass. All 200 measured
commands exceed the unchanged 100 ms target. Close this bounded comparison;
the result does not justify a rerun, cache activation or a general speed claim.

| Complete command | Baseline median | Candidate median | Baseline p95 | Candidate p95 |
| --- | ---: | ---: | ---: | ---: |
| Primer, 50 per arm | 156.28 ms | 160.35 ms | 850.42 ms | 365.98 ms |
| Successor, 50 per arm | 165.94 ms | 173.56 ms | 494.58 ms | 323.86 ms |
| All measured, 100 per arm | 158.15 ms | 164.34 ms | 544.34 ms | 337.10 ms |

The minimum measured command took 125.05 ms on baseline and 120.15 ms on
candidate. The median paired two-command round difference was +12.26 ms
(candidate minus baseline), favoring baseline. Primer and successor paired
median differences were -0.07 ms and +0.23 ms respectively. These are distinct
summaries; a difference between marginal medians is not a median paired change.

Earlier-event acknowledgement overlapped all 50 measured successors in each
arm and no measured primers. The intended contention was exercised. Intervals
cover the entire acknowledgement method, including lock waits, not isolated
COMMIT. Both primer and successor tails were lower on candidate, so attributing
the entire p95 difference to acknowledgement scheduling would be unsupported.
The deterministic scheduling regression still proves event-loop responsiveness
during acknowledgement I/O; it does not prove faster complete capture.

## Inputs and execution

The [prospective protocol](../../plans/2026-11-08-replay-ack-comparison-v1.md)
was frozen before execution. Baseline is canonical `7a77e40f`; candidate is
`ec4a36b2`. The [exact package pair](replay-ack-packages-v1.md) differs only in
mnemos/client.py and wheel RECORD. Both used the same buffered native binary,
SHA256 `b154c81345e4a7473086b946599baf2535daafc627133b1131984d7091914683`.
The frozen runner, original auditor and owner are from improvements `8f5ece12`.

There were five warmup rounds and fifty measured rounds, alternating AB/BA.
Both commands in every pair were timed through process exit, with no drain,
file write or response inspection between primer and successor. All 220
scheduled commands are retained and executed, including the 20 warmups. The
fixture used simple embeddings, unavailable cognition, disabled experimental
modules and separate disposable stores. Original Git resolution, live registry
checks, privacy enforcement and FULL SQLite capture commits remained active.

Each arm retained 110 unique durable events, 110 completed extraction jobs and
110 matching queued receipts. The corrected independent audit reopened both
databases, reconstructed original and queue IDs, checked content and scope,
verified empty queues and acknowledgement traces, and established barrier
ordering before the next arm and round. Both owners exited, their ports closed,
and the enclosing job recorded zero active descendants after cleanup.

Frozen package members, helpers, native binary, settings, registry, selected
dependencies and the seven previously observed Git runtime files were unchanged.
The Git manifest is a partial runtime binding, not dynamic loader closure or
cache approval. No live host configuration, pilot installation or original
dirty 0.7.0 source was changed.

## Preserved failures and audit correction

The first launch attempt, `.venv/replay-ack-comparison-v1`, stopped before
starting either owner or any scheduled command. Its complete environment hash
differed because CODEX_TOOL_CALL_ID changed between tool invocations. A
content-free per-key hash probe isolated that difference; direct and supervised
launches within one invocation matched. All 220 commands remain unexecuted in
that attempt, its failed audit is retained, and its job was cleaned up.

The measured run, `.venv/replay-ack-comparison-v2`, prepared and launched within
one invocation without changing helpers, environment checks, schedule or
thresholds. Its freeze SHA256 is
`973c3b0991a679772ca36f9403b454c396ee784b1a00d500214f3dbd23d85cc1`.
No measured command was retried and no measurement rerun occurred.

The frozen original auditor then failed because it expected the registry root
to equal the working directory. Canonical canonical_project_root returns Git's
normalized common directory, which is project/.git for this fixture. The
registry had that correct value before execution. A separate
`benchmarks/native_hosts/replay_ack_audit_v2.py` corrects this expectation and
records amendment hashes. The original helper and failed audit remain intact.
The corrected final audit passes the same raw evidence; no measurement,
identity, output or threshold changed. Independent review confirmed the scope
contract and recomputed the reported distributions.

Before freezing, 40 runner, auditor, owner and Windows-job tests passed. The
subsequent 23-test runner/auditor check includes the added prior-event overlap
regression. Four additional tests cover the corrected Git identity expectation.
Black and diff checks passed for the corresponding source changes. These are
tooling checks, separate from product, native-host and release acceptance.

## Evidence and remaining requirements

Private archive: `.venv/replay-ack-comparison-v1-evidence.zip`, 552 members plus
SHA256SUMS.json, 5,029,706 bytes. Every archived member was hash-verified.
Archive SHA256:
`54a378a22b3c6cd7e7e321c3ec9a980aee45ad2bea17645535747f5428136235`.
It includes both attempts, original and corrected audits, raw databases,
receipts, intervals, ledgers, helper/protocol/native/wheel inputs and test
receipts. Runtime dependency files remain external with recorded hash bindings.

Complete capture latency remains unmet. This synthetic fixture does not
establish real embedding-workload performance, 10,000-fact recall, semantic
support/abstention quality, current native-host acceptance, Desktop behavior,
pilot usage or release readiness. Those original requirements remain open;
the scheduling repair and its narrower deterministic guarantee do not replace
them.
