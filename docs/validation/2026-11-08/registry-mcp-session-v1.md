# Persistent MCP session respects external project access changes

External project disablement and re-enablement passed through one persistent
MCP connection on installed candidate `ec4a36b2`. This uses the real installed
MCP server and the default MnemosService factory. It is protocol integration
evidence, not an actual Claude or Codex host run.

The completed check made 16 tool calls. An explicit remembered fact was
consolidated, its job completed, and its source inspected. The controller then
disabled the project in the disposable registry without restarting the MCP
server or service. Seven calls were denied: full memory retrieval, compact
captured-evidence retrieval, listing, inspection, job status, another explicit
store and forgetting the original fact.

All denials returned scope_unavailable with committed=false and no fact/source
content. The store response also reported event_committed=false and the
expected scope. Counts and hashes matched before and after these calls for
memory_chunks, episodic_events, memory_jobs, fact_evidence, fact_revisions and
memory_suppressions. This proves no observed changes in those six durable
tables; it does not prove absence of transient writes or changes elsewhere.

After external re-enablement, inspection and recall returned the original fact
unchanged, and another explicit store completed. The reopened database contains
exactly two facts, two source events and two completed jobs, no blocked-write
marker and no suppression record. The same service PID, owner thread and token
remained in use. One ClientSession spans the transitions, and the MCP bootstrap
creates its startup record exclusively, supporting one server startup.

The owner exited, its port closed, and the enclosing Windows job reported zero
remaining descendants. Independent raw-record/database checks and read-only
review support this scoped result. No product source change was needed.

## Artifact and fixture

The wheel is the candidate from [the prepared package pair](replay-ack-packages-v1.md):
SHA256 `ff7e97e1ed24a7122dd4094b9dcdd4ed2c5ece3e805ea3becca58e0e62502d5d`.
A new private environment at `.venv/registry-mcp-runtime-v1` installs that wheel
with the 57 pinned package versions from the earlier native validation runtime.
MCP is 1.30.0. Dependency compatibility passed, all 98 installed package members
match the prepared artifact, and the final package check is unchanged.

The fixture uses simple embeddings and the default unavailable cognition
provider, with all experimental modules disabled. It exercises explicit memory;
it does not test semantic extraction quality. The frozen timing environment,
pilot installation, live databases, host configuration and original dirty
0.7.0 checkout were not changed. No model inference or credential copy occurred.

## Preserved setup and verifier failures

The final successful run is `.venv/registry-mcp-session-v5`. Earlier attempts
remain intact:

- v1 rejected the fixture's explicitly selected mock cognition provider during
  default-service initialization. Removing that setting uses the supported
  unavailable-provider path; the product guard was not bypassed.
- v2 started the default service but could not import the optional MCP package
  from the timing environment. A separate environment supplies MCP.
- v3 correctly denied the first revoked retrieval, but the verifier incorrectly
  required the outer MCP isError flag. This implementation returns the explicit
  service denial as structured tool content.
- v4 correctly denied the first five reads and the store, but the verifier's
  exact response schema omitted the store tool's event_committed and scope
  fields. The corrected check requires those fields to be uncommitted and match
  the expected project; it still rejects extra content and generic errors.

Eight focused verifier tests pass, including rejection of misleading success,
generic provider errors, leaked facts, wrong scope and a committed store event.
Black and diff checks pass. The earlier failures are fixture/verifier failures,
not evidence that revocation leaked content or committed a denied write.

Private evidence: `.venv/registry-mcp-session-v1-evidence.zip`, 172 members plus
SHA256SUMS.json, 681,348 bytes. All archived members were hash-verified.
Archive SHA256:
`491270b6fdd3e970df7f87c42a65d63417f862a324aa2b9e8fb5687b25ff2a5d`.
It retains all attempts, raw tool results, databases, snapshots, helper versions,
the candidate wheel, package bindings, audit and test receipts.

Actual native-host continuity remains the next validation step. This check
does not establish revocation during an already-running operation, full native
lifecycle acceptance, semantic quality, latency, pilot usage or release readiness.
