# Byte-packing change in ordinary retrieval

The linear byte-packing change does not establish a meaningful top-five service latency improvement. In a separate development measurement over 10,000 synthetic stored facts, service medians were 55.93 ms before and 56.87 ms after. Packing itself took roughly 0.05 ms in either arm. At 100 returned records, packing fell from 0.28 to 0.18 ms while service time remained about 91 ms. This limits the practical interpretation of the earlier large-context stress result.

The source was clean at `022b0ff0a1b41c2446ec50182289ad3c504c5c2d`. Both arms used that service implementation; only the packing function differed, with the old function loaded from `bc482489`. Separate SQLite stores started as copies of one 10,000-fact database. Scope, stored content, fixed two-dimensional embeddings, baseline profile and query were identical. Optional modules were disabled. No model or extraction was invoked.

Each combination had two warmups followed by twenty measured calls. Arm order alternated on each repetition. Top-five used a 1,200-byte budget; top-100 used 32,000 bytes. The timed service boundary includes retrieval and durable access accounting, but excludes startup, transport and host dispatch. Packing was instrumented identically in both arms. The schedule ran top-five before top-100; host load was uncontrolled.

| Returned records | Before service median | After service median | Before packing median | After packing median |
| --- | ---: | ---: | ---: | ---: |
| 5 | 55.93 ms | 56.87 ms | 0.0525 ms | 0.0537 ms |
| 100 | 91.86 ms | 91.01 ms | 0.2819 ms | 0.1775 ms |

All 80 measured selections matched by IDs, rendered context, byte count and budget count. The audit confirmed unchanged stored content, equal access counts and SQLite integrity in both arms. Each arm recorded 2,310 accesses including warmups. The first audit incorrectly read the legacy access field rather than the separate access table; the corrected audit joins that table. Both audit versions are preserved, and no retrieval measurement was rerun.

The [receipt](byte-retrieval-v1.json) binds the source, original snapshot, raw result and evidence archive. Archive SHA-256 is `c35862a9f62c899b2579df970485b7b805f6c93d673de3b0ae34a77805a7d37c`; raw records and helpers are in `.venv/byte-retrieval-v1-evidence.zip`. All processes completed. These are source-level synthetic service timings, not installed native-command or fresh quality acceptance. No precision, recall or supported-facts density was graded. The original quality, complete-command latency and pilot requirements remain open.

The result directs latency work back toward the complete command boundary and its measured costs. Further optimization of ordinary top-five byte packing is not supported by this measurement.
