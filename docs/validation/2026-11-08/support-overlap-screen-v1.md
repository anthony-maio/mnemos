# Support overlap feasibility screen

Concurrency alone is not a credible route to the 500 ms complete-retrieval target for the existing QA/contrast pair. Across 132 preserved observations, contrast was slower than QA every time. Assuming unchanged component durations, perfect overlap, and zero composition overhead, none would finish support selection within 500 ms.

| Quantity | Minimum | Median | p95 |
| --- | ---: | ---: | ---: |
| Measured sequential support | 1,553.74 ms | 2,990.80 ms | 3,656.76 ms |
| Measured QA component | 98.94 ms | 247.85 ms | 345.68 ms |
| Measured contrast component | 1,441.36 ms | 2,738.90 ms | 3,341.55 ms |
| Estimated perfect overlap | 1,441.36 ms | 2,738.90 ms | 3,341.55 ms |

The estimate is computed for each observation as `max(qa_ms, contrast_ms)`, then summarized. Percentiles use linear interpolation. No component percentiles were added or combined. Each of the three repetitions independently has zero observations within 500 ms under this estimate.

This is counterfactual arithmetic, not measured concurrency or a guaranteed performance bound. Component durations could change under overlap. It excludes retrieval, transport, and cleanup costs. The inputs are the existing 44 development questions repeated three times; they are not fresh quality evidence.

The audit verified the original private archive against its published receipt, then checked that its observation file exactly matched the retained working copy. It made no model calls, replays, threshold changes, or writes to the original evidence. The derived records preserve every observation ID and both measured component times.

The opt-in concurrency implementation remains isolated and tested, but no live timing experiment is justified for this unchanged pair solely to pursue the 500 ms gate. Further performance work must address contrast inference or its request representation, while independently preserving answer support, critical-fact retention, abstention, and source validation. Dropping contrast or weakening acceptance thresholds is not established as a solution by this screen. The failed semantic evaluations remain closed and unchanged.
