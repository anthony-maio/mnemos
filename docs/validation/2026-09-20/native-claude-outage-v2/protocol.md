# Native Claude service outage: raw capture recovery v2

Prospective successor to the [failed v1 attempt](../validation/2026-09-20/native-claude-outage-failure-v1/README.md).
Freeze this protocol, the v2 runner, and its imported acceptance helpers before
one native attempt. Preserve v1 and every failure; do not rerun v1 or replace
its output. This protocol tests the autostart and isolation repairs on product
`a64afb73d1b99510cfa32e96164bf6c11458b51e`.

The runner must require all six source/package checks and the installed
contract/replay checks to pass for that commit. It must verify the active
interpreter, wheel hash, installed Python package bytes, committed experiment
inputs, pinned Claude Code 2.1.191 executable, and plugin bytes retained from
the three-host v7 run. Record the source and installed verification-file hashes
before inference. A prerequisite failure permits no native inference.

Use only the fresh directory
`C:/Users/Ant/AppData/Local/Temp/mnemos-native-claude-outage-v2`.
Pin SQLite storage in both the disposable TOML and host environment. Pin
runtime, descriptor, registry, profile, project, routing review, checkpoint,
plugin, and Claude configuration paths inside that directory. Verify resolved
storage and the installed default client's disabled autostart setting before
launch. Do not touch live host configuration or clean the default database.

Start the injected real HTTP service with all five modules disabled, real
Nomic embeddings configured, and background processing stopped. Require its
authenticated dispatcher health response to identify the expected SQLite
database and stopped scheduler. Record the module settings supplied to each
actual injected engine. Stop the service and verify descriptor absence and
refusal of connections on its former port.

Run exactly one native Claude conversation with `claude-sonnet-4-6`, a $0.50
maximum budget and 180-second deadline. Use the original v1 prompt and marker.
Require the full native transcript to contain that exact user prompt and
`Acknowledged birch-window-618.` from the pinned model. Reject tool-use or
tool-result blocks anywhere in the native records, and require the transcript
session IDs to match the launched conversation. Require process completion,
copied-authentication removal, continued descriptor absence, and a nonempty
durable outbox. Do not retry inference or repair model output after a failure.

Restart the same disposable service and use an ordinary fresh client health
request to drain the outbox. Match native text identities to capture receipts
without counting provisional or duplicate deliveries as extra messages.
Independently verify raw event content, roles, physical scope/source/profile/
session fields, metadata, pending job identity, zero facts, and SQLite integrity.
Replay the saved payloads once, restart again, and require stable events/jobs,
the same physical source checks, and an empty outbox. Record health responses
at each start. No local embedding or cognition response may occur.

Success is `native_service_outage_raw_capture_observed`. Pending processing
jobs are expected; this is raw capture recovery, not completed fact extraction.
It does not establish provider-outage or in-flight crash recovery, Codex or
Hermes outage behavior, Desktop support, model quality, or daily-use reliability.
Stop the owned service and remove only copied authentication/runtime artifacts
through the existing cleanup paths. Preserve the full failed report if cleanup
or any acceptance check fails. Archive sanitized evidence; never include the
default database or its private backup in the research packet.
