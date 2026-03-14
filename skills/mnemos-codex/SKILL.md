---
name: mnemos-codex
description: Use when Codex users or Codex agents need to install, configure, validate, troubleshoot, or operate Mnemos through MCP, or when they mention Codex memory, AGENTS.md memory policy, Codex Automations, or Mnemos in Codex.
---

# Mnemos for Codex

Use this skill to put Mnemos on the supported Codex path without overstating host automation. In Codex, the practical Mnemos workflow is:

- recall at the start of substantial work
- curate durable facts during or after the task
- consolidate before finishing

## Default path

- Prefer `pip install "mnemos-memory[mcp]"` and `mnemos ui`.
- Treat the blessed Codex setup as two parts: MCP config plus a repo-level `AGENTS.md` memory block.
- Read `references/install.md` for install and validation.
- Read `references/operations.md` for daily use, troubleshooting, and honest capability framing.

## Claim discipline

- Safe to claim: Mnemos works in Codex through MCP, shared `MNEMOS_CONFIG_PATH`, repo-level `AGENTS.md`, and optional maintenance Automations.
- Do not claim built-in prompt/tool lifecycle hooks or Claude Code parity.
- Hard auto-capture in Codex is host-dependent and not shipped by Mnemos today.

## Daily loop

1. Start in recall mode: call `mnemos_retrieve` with a task-focused query and repo-scoped arguments.
2. Do the work, then switch to curator mode: store only durable facts with `mnemos_store`.
3. Use `mnemos_inspect` before storing a correction or when a retrieved memory looks suspicious.
4. Finish substantial work with `mnemos_consolidate`.
5. If Mnemos MCP tools are unavailable, continue normally instead of blocking work.

## Recall mode

- Use at the start of coding, debugging, review, or handoff tasks.
- Query for architecture, current repo conventions, recent fixes, environment quirks, and user preferences that matter for this task.
- In Codex, prefer `current_scope=project`, `scope_id=<workspace or repo name>`, and `allowed_scopes=project,global`.

## Curator mode

- Use during or near the end of a substantial task.
- Store decisions, constraints, environment facts, recurring bug patterns, and stable preferences.
- Skip one-off chatter, ephemeral plans, stack traces without reusable lessons, and secrets.

## Avoid

- Do not present Codex Automations as session capture.
- Do not tell users to type memories manually as the primary workflow.
- Do not market Codex as Tier 1 until real daily-use validation is complete.
