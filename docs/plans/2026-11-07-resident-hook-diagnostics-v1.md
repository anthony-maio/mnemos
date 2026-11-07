# Resident hook cost diagnosis

The installed comparison at `7d31ebd3` reduced median Stop and prompt time but
still failed the 100 ms complete-hook requirement. Preserve that run and its
original byte-comparison failures. This diagnostic run locates remaining cost;
its instrumented timings cannot replace the installed performance result.

Use the same resident-worker wheel, Python, dependencies and real embedding
cache in a new private C: runtime. Clone the pristine 10,000-fact seed database,
update only private configuration paths and registry root, and keep the accepted
project identity. Verify seed bytes and all installed/model bindings. Keep the
private launcher policy, checked fallback and normal maintenance behavior.
Record effective settings only as a digest or a whitelist without credentials.

Record five empty-command startup controls, then one warmup and five observed
complete commands each for Stop and UserPromptSubmit. Use unique source/session
identities and fixed synthetic text. Prompt handling must use real retrieval and
return complete context. Record all commands and failures, including warmups.

Add timing wrappers in the disposable entry scripts, without editing installed
product files. In the worker, measure the complete dispatch, effective-settings
check, project resolution, adapter capture/retrieval, outbox operations and HTTP
dispatch. In the memory service, measure dispatch, capture transaction, retrieval,
query embedding and storage search. Wrappers must call the original function
with unchanged arguments and results. Buffer metadata-only spans per request;
write them after dispatch. Label nested spans and do not add overlapping times.

The command probe measures script duration and the worker request. The enclosing
driver still measures complete process execution. Differences can locate costs,
but they are not independent latency gates or exact operating-system startup
measurements. No implementation change or thread-setting treatment belongs in
this initial run. A proposed optimization needs its own validation afterward.

Correlate worker and memory-service spans by synthetic session identity. Verify
one worker dispatch and one committed capture per real command, one real query
embedding per prompt, and the expected scoped accepted facts in every context.
Check all source/job records after shutdown and keep queued versus completed
status distinct. Confirm the accepted cohort and original seed are unchanged.

Supervise the memory service and worker with the existing Windows ownership
helper. Register their handles immediately and stop only verified owners. Keep
runtime descriptors and credentials out of the archive. Preserve instrumentation,
bindings, observations, spans, final database and aggregate diagnosis privately;
publish only the protocol, aggregate report and receipt. Keep the pilot inactive.
