# Live tokenizer integration: verified, acceptance still open

Live Qwen token counting works through canonical engine packing and the local
service response. All 408 scheduled observations completed without operational
failure. Across 136 matched groups, injected callback failure returned the same
facts, sources and context as ordinary byte fallback. All native contexts stayed
within their token budgets and detokenized exactly.

The same numeric budget represents different units in the two methods. At 64,
native counting retains all 40 expected facts, while the conservative byte arm
retains five. Native counting still returns 83 unsupported facts. This verifies
packing behavior and exposes the cost of conservative fallback; it does not
establish semantic support or a quality pass.

| Budget | Counter | Expected facts retained | Unsupported facts | Actual Qwen context tokens | Supported facts / 1,000 tokens | Local-service p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | Native | 40/40 | 83 | 2,859 | 13.99 | 347.94 |
| 64 | Bytes | 5/40 | 5 | 151 | 33.11 | 165.62 |
| 64 | Failed native -> bytes | 5/40 | 5 | 151 | 33.11 | 141.44 |
| 2048 | Native | 40/40 | 87 | 3,004 | 13.32 | 517.20 |
| 2048 | Bytes | 40/40 | 87 | 3,004 | 13.32 | 191.57 |
| 2048 | Failed native -> bytes | 40/40 | 87 | 3,004 | 13.32 | 192.92 |

Totals cover 68 known questions per row. Byte arms at budget 64 correctly
abstain on 24/28 absent-answer questions, but also lose 35 expected facts.
Native counting at 64 and all three arms at 2048 abstain correctly on 0/28.
Higher apparent density or abstention under a tiny byte budget cannot substitute
for recall. No fresh labels or critical-fact annotations were introduced.

## Measurement and invariants

The [protocol](../../plans/2026-11-08-live-tokenizer-retrieval-v1.md) was committed
before execution. Runner `cd19c84a` imported canonical product source
`c77933402a8bb4629211f24be22c3555fc872a76`, with 142 frozen files, 57 dependency
versions and 17 verified BGE assets. Each observation used its own complete
scenario database copy. Actual BGE query embedding, ranking, packing and native
callback work are inside the local-service timer. Database copying, integrity
checks and final-context tokenization/detokenization audits are outside it.
Embedding initialization and an unrelated query warmup are recorded separately.

The native counter made 254 uncached HTTP requests. Callback median was 1.49 ms,
p95 15.04 ms and total 1,185.71 ms. These component statistics are not additive
with retrieval percentiles. The 517.20 ms local-service p95 at budget 2048 already
exceeds 500 ms before client transport and host launch. Small scenario stores do
not establish the separate 10,000-fact requirement or explain timing differences
from earlier runs.

The audit reconstructs all native counts from raw replies and verifies returned
fact text, versions, scope IDs and evidence references against original SQLite
snapshots. Accepted data, original databases and frozen file bindings remain
unchanged. Ranking identities, scores and ordering match across each three-arm
group. Selected-result trace fields are checked against returned fact scores.

The first audit incorrectly compared those post-selection trace fields with
unselected rows, which contain only ranking fields. It stopped at that mismatch.
The retained corrected audit compares the complete common ranking fields and
separately verifies the selected fields. No observation or grade changed, and
no retrieval was repeated. SQLite integrity and foreign-key checks pass.

## Cleanup and limits

The model unloaded and both owned endpoints closed. The original process-tree
stop reported one unsupported termination; subsequent inspection found all seven
recorded process IDs absent. Controller and measurement exited with code zero.
Both the original cleanup output and follow-up verification are retained.

The [receipt](live-tokenizer-retrieval-v1-receipt.json) binds the verified
547-member private archive, SHA-256
`ab44f16569924cf0ac96013bbd00d587f86eb0192c5ac841a0df528ad5e23e1e`.
It retains observations, copied databases, original snapshots, sources, helpers,
runtime records and both audit versions. Large model assets remain separately
retained and hash-bound.

This establishes live local Qwen callback integration, including fallback. It
does not establish Codex or Claude tokenizer compatibility, whole host-envelope
budgeting, source-span compression, fresh quality or pilot use. Production code,
installed pilot artifacts, live configuration and publication remain unchanged.
