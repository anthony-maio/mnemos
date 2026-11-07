# Offline sizing for three support checks

The prospective prototype fits the current request limits over all 44 reused
development questions. It separates the requested value, its own source
support and conflicting comparison evidence into three typed choices. No model
was called, and no quality score changed.

The [receipt](support-three-checks-v5-preflight.json) records 22 cases, 27
answerable questions, 17 no-answer questions and 78 candidate requests. The
largest rendered prompt was 4,210 bytes, below 5,120. The largest per-query
typed request total was 9,591 bytes, below 64,000. The preflight included all
eligible fixture facts, checked the source firewall and complete-probability
schemas, and verified both dataset hashes remained unchanged.

The proposed combination requires the requested value to be present, one of
the candidate's own sources to support it, and no unresolved supported
comparison conflict. Every selected component must meet the existing 0.8
threshold. Provider failures receive no abstention credit. This combination
and its production dependency checks are not implemented by this preflight.

All six private archive members verified. The archive SHA-256 is
`79ab66064dc5ccf671a33399a4ee3e262cb474a75c6d230701f14717b482fbf2`.
The evidence establishes prompt size and schema construction only. Model
accuracy, calibration, latency, token counts, production integration and fresh
evaluation remain unmeasured.
