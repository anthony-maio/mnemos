# All-module provider recovery: preserved failure

This installed d937892 run resumed the frozen two-event outage database using
local Ollama, the production prompt and all five module flags enabled. The
prerequisite temperature comparison exited successfully before this run began.
The recovery runner exited 1; no automatic rerun was selected.

The assistant acknowledgement completed on attempt four with no durable facts.
The user event exhausted attempt five with MnemosExternalServiceError. Zero
facts were created. The all-jobs-completed assertion failed; scoped retrieval,
correction, restart, forgetting and subsequent replay phases did not run.
Source events and the terminal database are preserved. SQLite integrity and
foreign-key checks pass for the backup.

The provider error's underlying cause still needs diagnosis. This record does
not attribute the failure to the lease issue, infer a successful all-module
execution from enabled flags, or close T17/T18. The newer lease repair was not
part of this frozen installed run.
