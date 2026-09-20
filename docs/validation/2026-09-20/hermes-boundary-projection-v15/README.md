# Boundary projection repair and remaining evidence-write failures

Three new tests reproduce /new rejecting durable whitespace and multimodal
messages, and failing to reject a changed persisted tool name. The repair
accepts the actual persisted representation or the restored conversation
projection of the same row and compares tool_name. The original nine boundary
checks plus four new cases and existing CLI cases pass: 20 total.

The combined selection records 96 passes and three failures. The original
compression rotation failure remains. Although its fixture initially creates
the parent without cwd, the host can assign a working directory during setup
or rotation. Transcript-only handling for genuinely unscoped rows therefore
does not resolve the existing host regression. The earlier v12 description
of this failure as simply unscoped was incomplete.

Two new direct SessionDB cases expose separate evidence-write gaps. Recovery
of a blank assistant row stores its answer but yields no memory evidence.
Legitimate user-content expansion updates the transcript while leaving stale
evidence, so projection rejects it as a source mismatch. Both failures remain
open at this frozen candidate. Direct tampering must remain rejected when
these legitimate transactional update paths are repaired.

This is an isolated host candidate. Nothing here establishes live installation,
complete capture, service delivery, or general integration readiness.
