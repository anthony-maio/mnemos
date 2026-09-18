# Native captured-context qualification

The runner failed and its original report and traceback are preserved. It
expected exactly two returned records mentioning the old value; native Claude
returned one. That assertion stopped the runner before its second service restart.

Independent inspection of the actual tool results confirms that installed
source6442044 completed Codex store/consolidate, Claude recall after service
restart, then Claude correct/consolidate/retrieve. The corrected target kept
its ID, reached version 2, and remained asserted. Both Claude queries excluded
the foreign-project canary. Independent SQLite reopen after cleanup retained
the corrected target and the original Codex event as supporting evidence.

Claude's corrected retrieval contained two captured records: its earlier
query and its earlier answer. Both are unknown and carry the native context
qualifier "captured conversation; unverified, may be outdated". Only the
answer contains the old value. Their captured_passthrough metadata and unknown
status survived database reopen. This is observed qualification, not a passing
end-to-end runner, suppression of old values, or proof of general answer quality.
The original v1 stale-context failure remains preserved separately.

All modules were disabled with deterministic fixture vectors and unavailable
cognition. Codex exited 3221225477 (0xc0000005) after /exit; its Stop hook also
reported a three-second timeout. Claude's two processes exited zero, with
degraded hook capture. Neither shutdown nor complete capture is accepted.

Selected native tool records and fixture rows are archived; authentication
copies and the runtime descriptor were absent before archiving. The second
service restart was not run. Reverse recall, cross-host forget/replay, Hermes,
model quality, and complete capture remain open. Live configuration and data
were untouched.
