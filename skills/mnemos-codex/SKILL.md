---
name: mnemos-codex
description: Use when Codex users or Codex agents need to install, configure, validate, troubleshoot, or operate Mnemos through MCP, or when they mention Codex memory, AGENTS.md memory policy, Codex Automations, or Mnemos in Codex.
---

# Mnemos for Codex

Use this skill to put Mnemos on the supported Codex path without overstating host automation.

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

1. Call `mnemos_retrieve` at the start of substantial tasks.
2. Store only durable facts with `mnemos_store`.
3. Finish substantial work with `mnemos_consolidate`.
4. Use `mnemos_inspect` before storing a correction.
5. If Mnemos MCP tools are unavailable, continue normally instead of blocking work.

## Avoid

- Do not present Codex Automations as session capture.
- Do not tell users to type memories manually as the primary workflow.
- Do not market Codex as Tier 1 until real daily-use validation is complete.
