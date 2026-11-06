# Native explicit-memory checkpoint

This is one bounded Claude Code acceptance attempt for the November 6
improvement branch. It is not a daily-use pilot, a ranking-quality evaluation,
or a replacement for historical native results. Freeze this protocol and its
runner before inference; preserve any failed attempt without rewriting its
status or supplying a corrective model prompt.

## Binding and isolation

Product source is `24887ef3` and the installed wheel is
`mnemos_memory-0.7.0b1-py3-none-any.whl`, SHA-256
`bdc64792d918293ca1e01c482531c518e02c1aca9c5ccf6603e035a067ac8273`.
Build and Twine checks passed. The installed smoke receipt must match that
wheel and record successful CLI/MCP restart recall with real BGE embeddings.
The runner compares installed package bytes to the wheel, and source files
to the declared Git checkpoint. Its own helpers and protocol must be committed
before use; record their commit and byte hashes in the run manifest.

Use Claude Code 2.1.191 with `claude-sonnet-4-6`, observing that model in the
actual native stream. Use a fresh directory outside the checkout, distinct
Git repositories for the local and foreign projects, an isolated installation,
SQLite database, registry, service descriptor and plugin snapshot from the
product commit. All five cognition modules remain disabled; local FastEmbed
uses BAAI/bge-small-en-v1.5. There is no cognition-provider fallback.

The existing phase helper creates one disposable Claude home per phase and
copies authentication into it with private permissions. It removes that copy
after success, failure or timeout. Live host configuration and live memory
databases are outside this attempt. Launch background processes without a
visible window. Only the named Mnemos tools are allowed; shell tools, other
connectors and slash commands are disabled.

The service runs in a separate installed-Python process. A thin observer wraps
the default dispatcher's capture/remember/correct calls to retain original
payloads and receipts for replay. It leaves the production dispatcher and
scheduler settings intact. Native maintenance is allowed to finish explicit
jobs. Controller maintenance after a phase cannot substitute for the native
job-status evidence required below.

## Six phases

Each phase gets one prompt, a new native session, at most 180 seconds and a
$0.50 ceiling. The full attempt is bounded by six phases and $3.00. The CLI's
reported costs are observations, not a billing statement. No automatic retry
of a failed model phase is allowed.

Use the existing synthetic release-label scenario from the three-host contract.
Controller setup creates retained review, unrelated build and foreign-project
controls. Native actions must preserve those controls.

1. Teach one exact user fact with `mnemos_store`, request maintenance, then
   inspect the returned job ID with `mnemos_job_status`. A committed receipt
   alone is insufficient; native tool evidence must show that job completed.
2. Restart the service process. In a fresh native session, retrieve the fixed
   release query and inspect the exact asserted fact, version and source
   evidence. The original fact ID must survive.
3. Retrieve and inspect before correcting that ID at version 1. Request
   maintenance and inspect the actual correction job's completed status.
   Independent SQLite inspection must show the same ID at version 2.
4. Restart the service process again. Retrieve and inspect version 2 and its
   evidence in a fresh native session.
5. Retrieve and inspect before obtaining a related-forget preview. Review any
   additional selected facts and eligible sources, then apply the unchanged
   preview preconditions. Require the correlated apply receipt and inspect
   SQLite for absence of both labels in facts, source events, revisions,
   evidence and job payloads. Suppression records must remain; controls and
   graph integrity must survive.
6. Between forgetting and final native recall, restart, replay recorded original
   remember/capture payloads at their exact original identities, maintain and
   restart again. Every replay must be suppressed. In the fresh native session,
   query both the forgotten topic and retained review convention. Deleted
   labels must not reach that stream; the retained asserted fact must return.

Also query the foreign project independently and require its positive control
with the correct scope. Every native phase must have a successful process and
terminal stream result, matching session/model identities, correlated allowed
tool calls and unchanged controls. An assistant's claim of success is not a
substitute. Use the existing strict teach/correct/forget/replay acceptance
rules; correction permits a native maintenance call before status inspection.

## Evidence and limits

Retain the raw streams privately with prompt, process, normalized call and
database-state hashes. Record service PIDs across restarts. Always stop the
owned service, verify descriptor removal, and verify copied authentication is
absent. Cleanup failure fails the attempt.

Report native hook diagnostics separately. This attempt does not establish
complete automatic capture, extraction quality, supported-answer reliability,
native tokenizer compatibility, performance at scale, or daily-use readiness.
The pilot remains unstarted and has its own entry gates and authorization.
