# Optional learning pilot outcome policy

This policy is prospective for a newly numbered validation attempt. It does
not change the failed optional-native-v1 result. Use the observation reader's
complete call stream and verify the complete twelve-tool native inventory
before any tool dispatch, including discovery. Keep all raw observations.

Permit at most one call to each of the two built-in discovery helpers per
explicit phase, before any Mnemos call. Accept only empty arguments or an
explicit Mnemos server selector. Results must be successful listings containing
only Mnemos entries. Resource reads, other servers, pagination, repeated
discovery, additional helpers and unknown fields are outside this contract.

Require every Mnemos call to belong to the phase's existing operation set.
Reject unsuccessful results everywhere except the following optional absence
check in the forgetting phase:

- The call is inspection of exactly the target fact, with consistent ID aliases.
- A preceding related-forget call passed the existing complete inspection,
  preview, selection, precondition and committed-receipt checks for that target.
- The deletion completed before inspection started in the same native session.
  Review operations also completed before deletion started.
- The result is exactly the recorded schema-1, noncommitted, degraded
  `404 (fact_not_found)` response. Retain its unsuccessful outcome label.
  Other errors, statuses, extra payloads or failed mutations are rejected.

An absence response is optional, can occur at most once, and is not a successful
write or sufficient deletion evidence. Require the independent database absence,
suppression and retained-control checks, then exact original-source replay and
fresh-session/service-restart recall in the future runner. Original write job
completion, source inspection and project separation remain required.

Bind each observed call to its session, unique call ID and start/end sequences;
reject reordered, overlapping deletion/absence, mixed-session or uncorrelated
input. Do not infer causal ordering from list position alone.

Validate with deterministic negative cases and offline reconstruction of the
preserved trace. Passing this new phase policy offline does not pass the old
attempt or establish its unrun replay/restart checks. No model request, service
start, activation or pilot day belongs to this implementation increment.
