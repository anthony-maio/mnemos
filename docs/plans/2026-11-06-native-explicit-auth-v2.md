# Native explicit-memory authentication amendment

The first attempt remains failed at runner `c3a14fd3`. Native teaching passed,
including a completed job observation. The next fresh Claude session returned
`authentication_failed` before inference or memory tools. Cleanup passed.
The attempt must not be resumed or relabeled as a complete lifecycle.

The source credentials file contained an access expiry of November 1, 2026.
That metadata was read after the failed attempt; the credential contents were
not printed or changed. The normal CLI's cached auth-status command still
reported logged in. That is not proof that the copied login can authenticate
a new request. Token rotation in the first disposable session is a plausible
explanation, but the run did not retain credential hashes or refresh traces
that would prove it. Do not claim the user's normal login was validated or
damaged from this observation alone.

Before another attempt, use a separately authenticated validation profile.
Reject a missing, malformed, expired or short-lived access credential before
creating native phases or copying authentication. Require at least one hour
of remaining lifetime for the bounded six-phase attempt. Record only expiry
and the required lifetime, never token values. Do not refresh or replace the
user's normal credentials from this harness.

The next run uses a new output directory and a newly committed runner binding.
The six-phase lifecycle, model, budgets and acceptance criteria remain those
in the original protocol. Native failure must also mark the current phase
failed in the derived report; the original report's leftover running phase
label remains preserved and is superseded by its terminal failed process and
overall report, not rewritten.

A dedicated login can be created with `claude auth login --claudeai` under an
explicit `CLAUDE_CONFIG_DIR`. Authentication is interactive and must be completed
by the account holder. The current CLI help confirms those options; the
[official authentication guide](https://code.claude.com/docs/en/authentication)
describes separate configuration directories and sign-in behavior.
