# Mnemos for Codex

This unreleased checkout implements MCP tools and Codex lifecycle command hooks. Adapter tests pass; native conversation acceptance is tracked in the [compatibility matrix](client-compatibility-matrix.md). Verify generated configuration against the installed Codex version before treating automatic capture as active.

## Install and enable

Install this checkout into the Python environment that will run the hooks:

```powershell
python -m pip install -e ".[mcp]"
mnemos ui
mnemos enable-project --path . --profile anthony-local
$env:MNEMOS_USER_PROFILE_ID = "anthony-local"
```

Configure real cognition and embeddings in the canonical configuration. Prefer absolute paths. The shared service owns persistence; each Codex MCP process is a client. See [functional local memory](functional-local-memory.md) for providers, scope, maintenance, and retention.

The control plane previews Codex TOML changes before applying them. It generates an MCP entry and command hooks using the selected interpreter, preserving unrelated handlers. Review profile, project root, descriptor, and interpreter paths for the environment launching Codex. Saving Mnemos configuration requires a service restart.

## Automatic capture

The adapter accepts SessionStart, UserPromptSubmit, PreCompact, PostCompact, Stop, and SessionEnd. UserPromptSubmit and Stop capture role-tagged evidence with stable identities. SessionStart and UserPromptSubmit can inject scoped recall through additionalContext output. PostCompact does not inject unsupported fields. PreCompact and SessionEnd request bounded maintenance.

Capture commits evidence before acknowledging it; fact extraction happens later. A sanitized outbox can retain automatic events during an outage. Queued events are not reported as committed facts. Scope, identity, safety, and expected-version rules are enforced by the service.

See the [Codex hook reference](https://learn.chatgpt.com/docs/hooks) for the host contract. Installed-version behavior still needs the native smoke recorded in the compatibility matrix.

## Explicit memory and project instructions

The optional [mnemos-codex skill](../skills/mnemos-codex/SKILL.md) provides recall and curator workflows. Generate a repository AGENTS.md instruction block with:

```text
mnemos-cli antigravity codex --target codex-agents
```

Review older instruction packs before applying them. Scoped requests use the registry's stable project identity; omit scope_id when the client resolves it. Do not substitute the repository basename.

Use mnemos_retrieve to recall supported facts, mnemos_store for an explicit durable assertion, and mnemos_inspect to review evidence and revision history. Use mnemos_correct with the current expected_version to revise the same fact, and mnemos_forget with the current expected_version to suppress replay. mnemos_consolidate requests maintenance; mnemos_job_status distinguishes pending processing from completed facts.

Optional Codex Automations can schedule maintenance. They do not prove lifecycle capture and are unnecessary for the built-in maintenance scheduler.
