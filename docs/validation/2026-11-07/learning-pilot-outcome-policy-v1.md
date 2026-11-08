# Prospective learning-pilot outcome policy

The validator now distinguishes a narrowly expected post-delete inspection
response from a failed write or an outage. It also permits bounded, read-only
Mnemos resource discovery before memory operations. These rules apply to a
future native attempt; optional-native-v1 remains failed.

Every call must retain its unique native ID, session, model correlation and
start/end sequences. Discovery is limited to one call per supported helper,
empty or Mnemos-only arguments, and successful Mnemos-only listings. Calls to
other helpers or servers, pagination, repeats, and discovery that overlaps or
follows memory operations are rejected.

The only permitted unsuccessful outcome is an optional inspection of the
exact target after its reviewed deletion has completed in the same session.
The existing inspection, preview, selected-source closure, precondition and
committed-deletion checks still apply. The response must exactly match the
schema-1 noncommitted, degraded `404 (fact_not_found)` result. Its unsuccessful
label remains intact. Outages, failed mutations, different IDs, conflicting
aliases and inspection before deletion completes fail this policy.

The observation inventory API retains every call and verifies the complete
twelve-tool catalog before any dispatch. The existing strict inventory API
is unchanged. Phase-policy satisfaction is explicitly distinct from lifecycle
completion: original write-job completion, source inspection, independent
database checks, retained controls, source replay and restart recall remain
separate requirements.

Selected checks pass 251 cases: 110 outcome/deletion/review cases and 141
trace/inventory cases. Strict typing passes for both changed source modules;
all four changed Python files pass formatting. An initial typing check found
a missing name-type guard; it was corrected before the final checks. No full
product suite or new native acceptance run is claimed.

An offline regression reconstructs all 27 calls across the six explicit phases
of the preserved attempt and verifies six complete twelve-tool inventories.
The new phase policy accounts for two discovery calls and one expected
unsuccessful inspection in forgetting. It neither removes those observations
nor changes their outcomes. The original strict reader still rejects that
trace with its recorded error; the original run record and archive are intact.

The [receipt](learning-pilot-outcome-policy-v1-receipt.json) binds the source,
tests, prospective policy and private offline evidence. Fourteen watched live
paths and normal credentials remain unchanged. No model request, service start,
activation or actual pilot day occurred. Product source and package metadata
still match the frozen `f5a944ea` installation.

The next step is a newly numbered isolated native attempt with these rules
bound before inference, preserving the original prompts and project proposal.
It must complete the previously unrun source-replay and final restart-recall
checks. The original quality, timing and two-week pilot requirements stay open.
