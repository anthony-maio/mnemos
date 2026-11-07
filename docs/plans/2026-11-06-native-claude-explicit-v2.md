# Native Claude explicit-memory checkpoint v2

Run a new six-phase attempt after the account holder signs in to the separate
validation profile. The original failed attempt and its evidence remain frozen.
The authentication amendment's one-hour minimum access lifetime still applies;
metadata inspection alone does not establish that a native request will work.

Use product `929c44a44100592dd46e52b7f77b402f84beb9e3` and the installed wheel
`mnemos_memory-0.7.0b1-py3-none-any.whl`, SHA-256
`298b20ec0cc7dd1c348cc55db45a4774ef979642f491fca1ade93ee511175fc9`.
The [installed smoke receipt](../validation/2026-11-06/installed-smoke-929c44a4.json)
binds this artifact. It includes disabled experimental defaults and clarified
job-status guidance. This artifact binding supersedes the older product and
wheel in the original Claude protocol. It does not include the later temporal
answer-support experiment, which is outside production retrieval.

Keep Claude Code 2.1.191, `claude-sonnet-4-6`, the same synthetic release-label
scenario, one prompt per phase, six fresh native sessions, 180 seconds per
phase, a $0.50 per-phase ceiling and $3.00 total ceiling. No failed phase is
retried. The original teach, inspect, restart, correct, forget, exact-source
replay and retained/foreign-control acceptance rules remain unchanged.

The runner's current prompts identify the actual job ID returned by store or
correction. Consolidation reports aggregate maintenance, so it must not supply
a substitute job ID. Require native observation of completed processing.

Commit this amendment and its runner binding before inference. Use a new output
directory outside the checkout, the verified isolated installation, disposable
native profiles, copied validation authentication and a private observer log.
The account holder's normal profile is not a credential source for this attempt.
Record the validation source credential's hash before and after without exposing
its contents. Require unchanged source bytes and deletion of every phase copy.

Preserve all native prompts, streams and process receipts privately, including
failures. Verify the actual model, session and tool-call identities, SQLite
state, suppression on exact replay, owned-process cleanup and descriptor removal.
Publish only a sanitized summary and evidence hashes. This is a synthetic native
checkpoint, not either selected project's private daily-use pilot.
