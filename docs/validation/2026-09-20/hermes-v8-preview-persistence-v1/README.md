# Hermes v8 preview persistence diagnostic

Read-only inspection of the disposable native host databases confirms that
the successful forget previews in attempts 2 and 3 were saved intact. Each
parsed persisted preview exactly matches its earlier native tool result,
including the digest, candidate records, and source-event records. The
database bytes were unchanged by inspection.

Attempt 2 retained a 17,621-character preview with 10 candidates and 10
source events. Attempt 3 retained two 16,294-character previews with 9
candidates and 9 source events each. The selected sessions have no inactive,
compacted, or compressed-summary messages, and neither disposable home has
spillover files. The report records per-message state and finish reasons;
the raw preview strings are included separately.

This rules out a missing digest or missing candidate-source records in these
saved previews. It does not establish the exact outbound HTTP body: request
serialization and last-minute context processing were not recorded by v8.
The follow-up [request-observation protocol](../../../plans/2026-09-20-hermes-request-observation-protocol-v1.md)
addresses that remaining evidence gap without altering the failed cohort.

The source database hashes, trace hashes, read-only inspection script, and
selected synthetic preview messages are indexed in `evidence-index.json`.
Full host databases, configuration, and authentication were not exported.
The [v8 native failures](../hermes-native-cohort-v8/README.md) remain unchanged;
this diagnostic does not establish successful forgetting or model reliability.
