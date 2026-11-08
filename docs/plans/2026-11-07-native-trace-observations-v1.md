# Native trace observations before a new pilot attempt

The optional learning pilot's first attempt remains failed. Its frozen runner,
trace parser and raw evidence are retained in the private archive. Do not retry
its phases, rewrite its receipt or classify diagnostic reconstruction as a pass.

Add a separate observation API for the recorded Codex Code Mode format. Keep
the existing acceptance API strict. An observed operation is not necessarily
successful or permitted by a lifecycle contract.

The observation reader must preserve every supported dispatch. Distinguish
Mnemos MCP calls from the two built-in resource-discovery helpers. Require MCP
correlations for exactly the MCP calls; reject missing, additional, duplicate
or misplaced correlations. Bind both kinds to their actual recorded parent
cell, model call, runtime start/end, identical arguments and returned content.
Unknown helpers, namespaces, runtime failures and inconsistent records fail.

For Mnemos, retain a well-formed application result even when its status is
degraded or it contains an error. Label it unsuccessful; do not turn a missing
fact into success or drop the call. Keep structural, transport and content
consistency checks. The two discovery helpers require successful native results.
The existing sequence acceptance helper must reject unsuccessful observations.

Use deterministic synthetic regressions for both positive observations and
tampered identity, namespace, arguments, correlation, result and parent records.
Run the existing strict integrity cases against both readers. Then read the
original failed trace without inference and verify that observation retains
all nine calls: two built-ins and seven MCP calls, including the degraded
post-delete inspection. Confirm the strict reader still rejects that trace.

This increment does not define a new native acceptance policy. Before another
native attempt, write a prospective lifecycle contract for optional discovery
and expected missing-fact responses. That contract must bind an absence check
to a preceding reviewed deletion of the same fact, distinguish missing facts
from outages and rejected mutations, and retain restart/replay requirements.
The live proposal, package, host settings and zero-day pilot counters stay fixed.
