# Native Hermes v12: three failed lifecycles

The fixed three-attempt cohort completed zero full lifecycles on verified
source `5155ebddd8e8cf42df1c67f11a1578f5edfb43ec`, wheel SHA-256
`3828edfeab80f11a94a3b5519793af0b8b228b4bb5bef9ced34608186d6468e7`.
Runner and protocol were frozen at `b4909f4` before inference. Execution ran
from 10:20:31 to 10:27:12 UTC on September 20, 2026. All attempts established
cleanup; independent terminal inspection confirms unchanged local controls
and foreign-project facts, events, and evidence.

| Attempt | Terminal observation |
| --- | --- |
| 1 | Teaching, restarted recall, and correction passed. The corrected-recall phase returned no correlated native tool call. The corrected target remained at version 2. |
| 2 | Teaching, both restarted recalls, and correction passed. Forgetting stopped after preview without applying deletion. The target remained at version 2. |
| 3 | Teaching, both restarted recalls, and correction passed. An invalid selected source received the repaired 409 rejection. The model then applied an incomplete preview twice, received HTTP 500 twice, and fell back to target-only forget. That deletion committed, but the host exited 1 and the reviewed-copy lifecycle did not pass. |

No attempt reached source replay or final restarted recall. Attempt 3's
target is absent with four suppression rows; this does not establish erasure
of all conversational copies. Its initial selected source was a target source
outside the selectable `source_events` list. The later preview had limit 2 and
`complete: false`. Its digest and limit matched the repeated apply requests.
Storage rejects incomplete previews before selection validation, explaining
the observed HTTP 500 from that package. This is a source-grounded diagnosis;
the native trace contains no server traceback.

All 44 recorded model responses were checked against loaded-model observations
with the pinned Qwen3.5:9b alias, 65,536-token context, and positive GPU residency.
Exact outgoing requests and raw responses are preserved. Embeddings used real
Nomic. All five cognition modules were disabled for this transport baseline.
This is not full-module quality or daily-use acceptance.

The fixed invalid-selection rejection was observed, but successful recovery
was not. The separate client outbox defect found during this run is reproduced
by offline tests with a valid pending capture; do not attribute native failures
to that defect without evidence. This cohort used the unmodified installed
5155 package and does not validate subsequent client or rejection repairs.

`boundary-analysis.json` binds requests and loaded-model observations.
`terminal-analysis.json` records read-only final SQLite inspection.
`selection-analysis.json` binds the incomplete-preview diagnosis to trace bytes.
The evidence index binds raw files; this README is outside that index.
The [import preflight](../hermes-native-discovery-preflight-v12/README.md)
passed separately. Live host configuration, source, and memory were unchanged.
All earlier failures and open T17/T18 gates remain unchanged.
