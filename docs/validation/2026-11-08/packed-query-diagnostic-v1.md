# Frozen packed-query failure diagnosis

The failed packed probe received the intended inputs. An offline audit read the hash-chained request/reply journal and verified all 16 recorded model inputs against their owner snapshots: query, source text, fact text, annotation and claim order. No model calls, reruns or repaired scores were used.

Splitting the original semantic judgments exposes failures beyond claim identifiers:

| Observation | Result |
| --- | ---: |
| Correct relation judgments | 13/28 |
| Correct condition judgments | 11/28 |
| Unconditional claims assigned applies | 10/25 |
| Reordered pairs preserving judgments by fact | 0/3 |
| Reordered pairs preserving temporal intent | 1/3 |

The three pairs ask the same question over the same two facts in opposite order. Both owner and region pairs changed temporal intent as well as fact judgments. The conflict pair retained temporal intent but changed judgments. Seven of fourteen semantic cases returned the identical `D20261108|MAMX` string despite different questions. This is observed repetition, not proof of a particular runtime or model defect.

The original 7/28 fully correct two-part claim judgments and zero exact semantic cases remain unchanged. The two forced-support controls are excluded from the figures above. Output formatting success does not establish understanding of the codes, and exact-source quotation checks do not validate the query's relation or condition judgment.

## Consequence for the next candidate

Changing claim identifiers alone is insufficient. A subsequent development design should expose the meaning of each judgment directly, separate relation and condition decisions, and verify association under reordered candidates before testing full selection. It must retain unresolved, conflicting and time-qualified evidence. Larger explicit outputs may fail the latency requirement; accuracy improvement cannot waive that cost. The current evidence does not isolate encoding, prompt, grammar, runtime or model as the cause, so no causal improvement claim is justified without a separately frozen prospective comparison.

This diagnosis does not authorize tuning or rerunning the closed cohort. Source annotation quality remains independently unresolved. A future query screen must keep supplied annotations distinct from extraction and use separate development cases; final acceptance still requires fresh data and complete latency measurement.

The [receipt](packed-query-diagnostic-v1.json) retains every original/expected judgment and reordered-pair mapping. Original journal SHA-256 remains `7bace07fb5c68f0629b775ecb48599c74c020c7b647efbbb086da9a12f6b6571`; original grading SHA-256 remains `59ddedbad6e1a9e8070bf5ccf57be950cc5d6ba6262e805dca58ccd6c0569955`. The diagnostic helper is `.venv/diagnose-packed-query-v1.py`. No production source, package, host configuration or pilot state changed.
