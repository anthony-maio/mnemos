# Reviewed forgetting: local model development comparison

This is a three-case-per-model development diagnostic using the installed
`e454a85` wheel, not a native Hermes lifecycle or a T18 replacement. Qwen ran
first, then GPT-OSS. Each arm ran every scheduled case once. The raw reports,
SQLite fixtures, exact prompts and requests, tool traces, state snapshots,
module origins, package hashes, input hashes, and cleanup outcomes are kept.
Transport IDs are normalized for this direct Ollama runner; they are not
native Hermes identities.

| Model | Case | Status | Seconds | Failure or outcome |
| --- | --- | --- | --- | --- |
| qwen3.5:9b | backup-window | failed | 60.73 | TimeoutError |
| qwen3.5:9b | deployment-lane | passed | 64.33 | all gates passed |
| qwen3.5:9b | archive-bucket | passed | 33.53 | all gates passed |
| gpt-oss:20b | backup-window | failed | 60.65 | TimeoutError |
| gpt-oss:20b | deployment-lane | failed | 60.81 | TimeoutError |
| gpt-oss:20b | archive-bucket | failed | 26.72 | forget_phase_verified, reviewed_persisted, scoped_markers_absent, all_copies_selected, required_sources_selected, known_facts_absent, known_sources_absent, known_facts_suppressed, known_sources_suppressed |

Qwen's backup-window case and GPT-OSS's backup-window and deployment-lane
cases returned no model response before the 60-second request limit. They
therefore do not establish tool-selection performance. Preserved server-log
observations record canceled model loading, followed by later successful
loads. The old preserved tag also produces metadata errors during background
listing; those messages alone do not establish the cause of load latency.

Qwen's two completed cases passed every gate before and after shutdown.
GPT-OSS's archive-bucket case retrieved, previewed, and applied target-only
deletion with empty additional-fact and source selections. It made no
inspection calls, retained all three known conversational copies, and then
claimed that all related copies were removed. Its storage result and final
language disagree. Qwen's passing final responses also repeat deleted values
and IDs; since capture is disabled here, these passes do not establish that
native capture would avoid recreating forgotten material.

Both arms use actual authenticated service dispatch and synthetic seeded
facts with SimpleEmbeddingProvider. The five cognition modules are disabled.
Conversation capture is disabled. This does not measure extraction quality,
Nomic retrieval, source replay, restarts, native callbacks, WSL transport,
cross-host continuity, or daily use. Fixture IDs and paths differ between
arms. Three cases do not estimate reliability or establish a causal effect
of model size, reasoning effort, or architecture.

Offline runner development initially failed three cleanup checks because
the helper read a connection after the service had closed it. The repaired
helper reopens the database read-only; all four runner checks then passed.
Both XML reports are retained. The final offline report exercises scoring
and transport evidence separately from all real-model outcomes above.

The [prospective protocol](../../../plans/2026-09-19-local-model-forget-development-v1.md)
and [frozen profiles](../../../plans/2026-09-19-local-model-forget-profiles-v1.json)
define inputs and limits. The [readiness archive](../local-model-readiness-v1/README.md)
preserves the old model failure, successful replacement, and failed readiness
assertion. The earlier [native v6 cohort](../hermes-native-cohort-v6/README.md)
remains zero of three complete lifecycles on its original runtime. No result
here changes those outcomes or closes native acceptance.
