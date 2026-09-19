# Installed Hermes API outage and restart recovery

The [frozen protocol](../../../plans/2026-09-19-hermes-api-recovery-protocol-v1.md)
passed on September 19 using Mnemos source
`5e4b057b16906c2ec7dd302ae80748ef1ac5eddc`, wheel SHA-256
`7eaba268fb91ffae7d59b222cf8addb49d8b4b4857a37552e5404422e4009625`,
and installed Hermes `14a346345486418aa85ed6526549412ee2665ade`.
The coordinator completed in 16.778 seconds. Three separate WSL host processes
used Hermes MemoryManager and the actual Windows Mnemos bridge.

The service was stopped, and a deliberately invalid isolated configuration
prevented automatic starts from reaching live settings. Two transcript
messages survived host exit in the production Windows outbox while the fixture
database had no Hermes events. The required compression checkpoint returned
success backed by those two durable payloads. It acknowledged local buffering,
not service commitment or fact extraction.

After explicit service restoration, the second host's ordinary retrieval
request drained the outbox before any transcript resubmission. The retrieval
itself returned no facts: draining evidence does not retroactively change the
result of the request that triggered it. Subsequent callbacks and maintenance
left exactly two source events, two completed extraction jobs, and one fact
evidence link per event. Original occurrence IDs and timestamps matched
exactly across host processes.

The controller forgot the fact backed by the user event, using its exact ID
and version. After another service restart and a third host process replayed
the transcript, that fact and its evidence remained absent, the source
suppression remained present, and no replacement extraction job appeared.
The unrelated project control and foreign-project control retained their
IDs, content, and versions. Project retrieval did not expose the foreign fact.
All three host processes exited zero with no lifecycle diagnostics.

This is a bounded installed API observation with fixture embeddings, disabled
modules, unavailable cognition, explicit service restoration, a two-second
bridge deadline, and a 180-second host deadline. It does not establish automatic
daemon restoration, arbitrary hard-kill timing, full model-driven Hermes
lifecycle, native CLI cwd propagation, semantic quality, or pilot reliability.
No live host configuration, source, or memory data changed. The existing
CLI compatibility patch and model-turn evidence remain separate.

The report binds source, wheel, host revision, protocol, and executed helper.
`evidence-index.json` records 16 raw files, including host reports and logs,
executed helpers, the exact protocol, and selected final synthetic records.
Runtime descriptors, tokens, credentials, and databases are omitted. The
unexecuted draft was corrected during review before this single observation;
it is not a prior failed experiment or an additional passing run.
