# Native trace observation reader

A separate Codex observation reader now retains successful built-in discovery
calls and unsuccessful Mnemos application results. Each call carries its
dispatch kind and outcome. The existing acceptance reader remains strict, and
sequence acceptance rejects unsuccessful observations.

Both readers retain the model-to-cell, runtime identity, argument, result,
ordering and source-hash checks. Observation requires MCP correlation records
for exactly the MCP calls. The two supported built-ins must use the recorded
Codex runtime identity and matching successful content. Unknown helpers,
additional correlations, failed transports and malformed records are rejected.

The selected trace and sequence tests pass 93 cases; the inventory tests pass
34 more. These 127 checks include the original integrity mutations against
both readers, discovery tampering and retained degraded/rejected/error results.
Strict typing passes for the changed source module, and both changed Python
files pass formatting. This is selected validator coverage, not a full product
suite or native acceptance run.

An offline audit of the preserved failed forgetting trace reconstructs all
nine calls: two discovery helpers and seven Mnemos operations. The post-delete
inspection remains explicitly unsuccessful, with its original degraded
`404 (fact_not_found)` result. All calls retain their recorded model parents.
The strict reader still rejects the same trace with the original correlation
error. Raw evidence and the original failure archive are unchanged.

The [receipt](native-trace-observations-v1-receipt.json) binds source, tests,
test reports and private diagnostic evidence. No model request or service
start occurred. Fourteen watched live paths and normal credentials remained
unchanged. The installed product was not changed, and both pilot counters
remain zero.

This reader records evidence; it does not decide that discovery is authorized
or that a missing fact proves a successful deletion. A prospective lifecycle
contract still needs to connect an expected absence response to the preceding
reviewed deletion of the same fact, reject outages and failed mutations, and
retain the restart/replay checks before another native attempt.
