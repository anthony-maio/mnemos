# Claude scope identity repair

Claude capture identities now bind the profile, project, workspace, session,
source reference, role, and captured text. A native event ID reused in another
project no longer collides with the first project's capture. Checkpoint paths
and their contents also carry the resolved scope; offsets cannot silently move
between projects. Automatic transcript captures remain project-scoped and accept
only user or assistant roles with a nonblank source reference. Explicit memory
operations retain their existing project, workspace, and global scope controls.

The service derives the previous event identity from a verified capture. It
checks the old source suppression before accepting a new identity. An exact
old capture returns its existing job, including after processed source retention
has removed the raw event. A different project's old capture does not count as
a duplicate. No live database or checkpoint migration is performed. New captures include an
explicit identity-version field. Older services reject that unknown field,
leaving the client to retain the event in its durable outbox until the service
is upgraded; they cannot silently accept the new ID without its suppression
checks. No service restart or live upgrade is performed by this change.

Old source suppressions do not contain project ownership once their source has
been erased. The service therefore conservatively suppresses a matching legacy
identity within the same profile, even if the new request names another project.
It returns `legacy_source_suppressed`. Keep that suppression. If the information
must be reintroduced, use the explicit remember operation after reviewing the
intended project; do not remove tombstones to make automatic replay pass.

A legacy checkpoint without a scope binding returns a degraded result and stays
unchanged. Start a new native Claude session with a fresh session ID to capture
new conversation under a new scoped checkpoint. Do not copy the old offset into
the new file or import the old transcript as new conversation. Continuing the
old session's checkpoint requires a separately reviewed binding or migration;
this repair does not provide that migration command.

Redaction presents another identity boundary. Old Claude IDs were computed from
text before the client redacted it. Once that text is erased, a redacted replay
cannot reliably reproduce its old identity. Captures whose content or source
reference needs redaction, or already contains a redaction marker, fail closed
with a degraded diagnostic. They do not create a new source ID or advance the
transcript offset. A blocked transcript line therefore also delays later lines;
complete automatic transcript coverage is not claimed. Explicit remember can
save a reviewed fact through the existing write firewall. Never disable the
firewall to bypass this boundary.

Missing stable native IDs remain a separate coverage limitation. A fallback
content identity can deduplicate a retry, but cannot distinguish two independent
identical messages. The adapter reports degraded identity coverage when it uses
that fallback. It does not invent a counter that would change on replay.

Validation is recorded separately from the earlier native d17 package proof.
The new service regression tests cover project/profile separation, exact legacy
replay, retained-job deduplication, source erasure and restart, arbitrary alias
attempts, redaction ambiguity, and whitespace variants of forgotten sources.
The frozen b7 real-model evaluation remains bound to its original implementation;
these host identity changes are not part of that experiment.
