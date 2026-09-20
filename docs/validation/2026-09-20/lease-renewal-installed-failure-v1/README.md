# Lease renewal installed-package check: preserved failure

The isolated installed suite completed with 832 passes, four skips and one
failure in generic-stdio MCP startup. Its registry contains the checkout path;
a successful Git common-directory lookup identifies that same checkout by the
shared Git directory instead. Startup failed with Project is not enabled.
The existing canonical_project_root silently falls back to the filesystem path
when Git times out or fails. A separate injected-timeout reproduction demonstrates
this identity drift, but the original test did not log its Git result, so its
specific trigger is not established.

All new lease cases passed, but this installed checkpoint is not green. The
runner stopped at the failed contract probe; final replay validation did not
run. The frozen source checkpoint remains a separate passing result. Preserve
this failure before changing resolver behavior or running a new checkpoint.
