# Linear byte-budget context packing

The canonical delivery implementation now counts each rendered record's UTF-8 bytes once and joins accepted records at the end. Previously, the byte fallback concatenated and encoded the growing context for every candidate. That repeated work grew quadratically with selected context size. Host tokenizer counting remains on complete concatenations because its token counts need not be additive.

Selection still preserves ranking, whole records, qualifiers and exact budget boundaries. A failed host counter still restarts selection from all ranked candidates. The change does not select additional facts or establish that retrieved facts support an answer.

All 35 context-budget, host-budget, retrieval-contract and answer-support-context tests passed in 5.52 seconds. Black and diff checks passed. A separate fixed-seed differential audit compared 500 selections with the previous implementation at `bc482489`: Unicode, multiline content, qualifications, varying budgets, empty sets, top-k limits and invalid host counters. All returned fields matched, including omitted IDs and fallback metadata.

The synthetic packing measurement used five sequential repetitions at each size, before and after the change. Every selected context hash, record count and byte count matched.

| Records packed | UTF-8 bytes | Before median | After median |
| --- | ---: | ---: | ---: |
| 1,000 | 68,890 | 15.08 ms | 0.67 ms |
| 5,000 | 348,890 | 538.28 ms | 4.29 ms |
| 10,000 | 698,890 | 7,335.51 ms | 12.91 ms |

These are stress measurements of packing alone, with large budgets that retain every record. They are not ordinary top-five retrieval timings or the original warm 10,000-fact acceptance test. Construction and correctness assertions were outside the timed interval. The after measurement ran alongside the regression suite, and host load was not controlled. No semantic quality, supported-facts density, capture latency or end-to-end retrieval acceptance is claimed.

The [receipt](byte-packing-v1.json) retains all timings, source hashes and context hashes. Helpers and raw measurements are `.venv/measure-byte-packing-v1.py`, `.venv/audit-byte-packing-v1.py`, and `.venv/byte-packing-{before,after}-v1.json`; JUnit is `.venv/byte-packing-v1.xml`. The frozen packages, evaluations and live hosts remain unchanged. A new package would be required to deliver this source change.
