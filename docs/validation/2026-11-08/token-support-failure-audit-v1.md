# Binary support failure audit

Read-only inspection of the closed run confirms that its three failed questions follow the declared labels. No label change, model call, threshold change, or rerun was made. The private audit verified the archived evidence hash and the original response, outcome, and grading hashes before inspecting saved decisions.

| Case | Source and query | Saved decision | Finding |
| --- | --- | --- | --- |
| Partial source coverage | Fact says every region; source covers only east. Query asks about east. | A, 0.9999150038 | The selected fact overstates its own source even though a narrower answer is available. |
| Missing reason | Query asks why the port changed; source states the old and new ports and date. | A, 1.0 | The source contains no reason. A related event is not the requested answer. |
| Unsupported replacement peer | Current tar fact has a direct source. Peer claims replacement by zip, but its source says proposal with no decision. | B for both facts | The current fact was missed; the proposed replacement was correctly rejected. |

These outputs are consistent with source-coverage, requested-relation, and comparison-evidence failures. They do not expose the model's internal reasoning or establish a causal explanation. All three requirements were already explicit in the prompt.

There is a stronger conclusion about threshold changes. Under the existing rule, a generated A is necessary. The missed critical fact generated B, so its recovery is impossible by changing only the acceptance threshold. Critical retention on these saved outputs is therefore at most 11/12, below 95%. The false reason answer generated A with reported probability exactly 1.0, so every allowed threshold from zero through one retains it. This rules out a threshold-only repair of these outputs without searching for a better score on the closed cohort. It does not prove that another model or mechanism cannot solve the task.

Earlier checked-v5 work already separated requested value, own-source support, and comparison validity inside query-time model judgments. Merely adding those questions again would repeat that direction. The next development design should test query-independent source qualification bound to exact fact/source revisions, with all comparison evidence still visible. It must also address requested relations explicitly; source qualification alone cannot fix the false reason answer. Any saved query-time work must be measured against qualification, invalidation, and cache-miss costs.

The executable audit and its result are retained under `.venv/audit-token-support-failures-v1.py` and `.venv/token-support-failure-audit-v1/result.json`. The original run remains closed and experimental.
