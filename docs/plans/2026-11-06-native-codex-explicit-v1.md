# Native Codex explicit-memory checkpoint

Status: prospective acceptance protocol. Freeze this file and its helpers before
inference. This is a new Codex attempt; the failed Claude attempt and its pending
dedicated login remain unchanged.

## Binding

Use product source `6d46aa1afda6ab48d914caf0a51b558bc4733438` and the newly built
`mnemos_memory-0.7.0b1-py3-none-any.whl`, SHA-256
`e0874f18f3df5b9605bca19ba9bf0553ce56ddf8fbaba946fee2bc09e07b72ce`.
The wheel was built from its source distribution using isolated Hatchling 1.33.0.
Both artifacts passed Twine. Require a successful installed smoke receipt
binding that exact wheel before native execution. Verify the installed package
bytes against the wheel and product source, and record the runner commit and
all acceptance input hashes. The installed environment must be outside the
checkout and use real BGE small English v1.5 embeddings.

Use Codex CLI 0.146.0 through its native Windows executable and `codex exec
--json`, with `gpt-6-astra` and high reasoning effort. This model is present in
the current local host catalog and is the normal profile's configured model;
the actual native session must still prove the observed model identity.

The normal Codex file currently contains unexpired access-token metadata. That
alone is not proof of a working login. Before the attempt and each phase, require
a ChatGPT credential with at least one hour remaining, recording only its expiry
metadata. Copy it into the phase's protected disposable Codex home and force
file-backed credential storage there. Do not refresh, replace or write the
normal profile. Remove each copied credential in a finally block on success,
failure or timeout; verify no copied credential survives the whole attempt.

## Native execution and isolation

Use a fresh output directory outside the checkout, a fresh native home for each
phase, distinct local/foreign Git projects, an isolated registry and SQLite
database, and a separately installed Mnemos service process. Keep all five
experimental modules disabled and use the baseline retrieval profile. Do not
change live configuration or live memory.

Each phase gets exactly one prompt and at most 180 seconds. There are six
phases, no automatic retries and no corrective follow-up prompts. Codex exec
does not expose the Claude runner's dollar ceiling; this attempt is bounded by
the six single-turn submissions and their deadlines. Record provider usage
reported by the native CLI without claiming it is a billing statement.

Run background processes without a visible window. Disable shell execution,
code-mode host routing, apps, extra agents, memories, goals, plugins, browser,
computer, image and artifact tools, workspace dependencies, skill search and
automatic hooks in the disposable invocation. Set web search to disabled.
Expose only the Mnemos tools allowed for that phase through the MCP server's
`enabled_tools` setting. The native trace must contain only the allowed,
directly correlated Mnemos tool calls. Do not accept inferred correlation from
JavaScript text or an assistant's success claim.

The generated integration still uses the installed production MCP entrypoint
and the normal service dispatcher and scheduler. The controller records original
remember/correct payloads and receipts for exact replay. Controller maintenance
cannot replace the required native observation of a completed job. Automatic
capture and hook behavior are outside this explicit-memory attempt.

## Acceptance

Retain all six phases and the same lifecycle checks from the
[original checkpoint](2026-11-06-native-explicit-memory.md): native teaching with
completed-job observation; restart recall and evidence inspection; correction
of the inspected ID and version with completed-job observation; another restart
and version-two recall; reviewed related forgetting; and absence after restart,
exact original-source replay, maintenance and another restart. Preserve the
retained and unrelated controls, verify the foreign-project positive control,
and inspect facts, source events, revisions, evidence, jobs, suppressions and
graph integrity independently.

For every phase, require successful process exit, exactly one successful final
turn in stdout, a nonempty final assistant message, matching thread/session IDs,
one retained session, the pinned model, the exact single user prompt, and
correlated allowed MCP calls. A zero exit code alone is insufficient. Verify
native source evidence inspection and original fact identity across versions.
The final fresh session must not contain either forgotten label.

Freeze the first failure without restarting that attempt. Retain stdout,
stderr, private session traces, normalized calls, prompts, service PIDs and
database-state hashes. Always stop the owned service, verify descriptor removal
and verify credential cleanup. Cleanup failure fails acceptance.

This attempt does not establish automatic capture, extraction quality, supported
answer selection, host tokenizer compatibility, performance at scale or pilot
readiness. The two-week/two-project pilot remains unstarted and retains its
separate entry gates and live-configuration approval requirement.
