# Native Hermes v10: one complete lifecycle in three attempts

The frozen three-attempt cohort completed one full lifecycle on the verified
`f2f7414` package. Attempts 1 and 2 failed; attempt 3 passed every scheduled
phase. This is a bounded demonstration, not a reliability estimate or final
local-memory acceptance. No attempts or failed phases were repeated.

| Attempt | Result | Observation |
| --- | --- | --- |
| 1 | Failed at restarted recall | Teaching passed. The model then announced retrieval without making a tool call. |
| 2 | Failed at reviewed forgetting | Teaching, restarted recall, correction, and corrected recall passed. Deletion committed, but two of the nine selected additional facts had not been individually inspected. The reviewed-forget contract therefore failed. |
| 3 | Passed | Teaching, restarted recall, correction retaining the original ID at version 2, restarted corrected recall, reviewed deletion, source replay suppression, and restarted final recall all passed. |

Attempt 2's deletion receipt does not override the failed review requirement.
The eight inspection results covered the target and seven additional facts;
the apply selected nine additional facts. The storage absence and suppression
checks passed, but the evaluator rejected the incomplete review and did not
run dependent replay or final recall. Attempt 3 reviewed its selection and
passed those later checks. Both forget processes exited normally, in 120.066
and 50.148 seconds respectively; neither hit the outer deadline.

No stale-preview rejection occurred in this cohort. The successful attempt
therefore does not demonstrate recovery from the new typed rejection or prove
that the error-contract repair caused the improvement. Fresh identifiers and
captured conversations vary between attempts. The original v9 failures and
the separate installed-package stale-preview regression retain their results.

## Evidence and bindings

[Boundary analysis](boundary-analysis.json) verifies recorded request hashes
and byte counts, preview transport, and all 40 model-response profile checks.
Every check reported the pinned alias/digest, 65,536 context tokens, and
8,044,454,869 VRAM bytes. All three readiness checks and cleanup checks passed.
Both forget previews reached later outgoing model requests intact.

- Runtime source: `f2f741443e60f838b392ae8ea1d1a0b4d8c6c1f2`.
- Wheel SHA-256: `8552345d2c17c4c037347961eb3043d13b20f790f54a9ccc86a096add6197b0f`.
- Runner commit: `11756832908a7c7fc7e3285d7918426683182777`.
- Model: `mnemos-native-qwen35-9b:greedy-ctx64k-v2`, digest
  `87bc5156d86918e0435431ab0286167a4c79c4a4a259b82fba5f3f75022006e2`.
- Isolated Hermes source: `14a346345486418aa85ed6526549412ee2665ade`,
  with the unchanged cwd-only compatibility patch and checked import origins.
- Shared interpreter installation: `ebe3fcb2f85ebdbe0e469e9d295636ac930246b4`,
  with its dependency inventory checked before and after each attempt.
- Local Ollama and real Nomic query embeddings. All five cognition modules
  were disabled for this transport/lifecycle baseline.

The [protocol](../../../plans/2026-09-20-hermes-native-lifecycle-protocol-v10.md)
and inputs were committed before inference. The
[import-only preflight](../hermes-native-discovery-preflight-v10/README.md)
discovered eight tools; 123 offline harness checks passed. The cohort ran from
06:53:26 to 07:00:45 UTC on September 20, 2026. The evidence index binds 336
raw artifacts; this README is outside that index.

Source replay is scripted API evidence. Native actions before and after it
remain separately recorded. This experiment changed no live host source,
configuration, or memory and published nothing. The isolated patched-host
pass is not evidence that an unmodified live Hermes installation works.

Three-host continuity, complete capture, independent module quality,
Codex Desktop, and the operational pilot remain open. One passing attempt
does not supersede the negative T18 comparisons or establish daily-use
readiness. It satisfies the preceding bounded Hermes demonstration needed
to prepare the next three-host experiment.
