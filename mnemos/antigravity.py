"""
mnemos/antigravity.py — Shared host instruction artifacts for soft-auto Mnemos use.
"""

from __future__ import annotations

from typing import Literal

AntigravityHost = Literal["cursor", "generic-mcp", "codex"]
AntigravityTarget = Literal["policy", "cursor-rule", "codex-agents", "codex-automation"]

ANTIGRAVITY_HOST_CHOICES: tuple[AntigravityHost, ...] = ("cursor", "generic-mcp", "codex")
ANTIGRAVITY_TARGET_CHOICES: tuple[AntigravityTarget, ...] = (
    "policy",
    "cursor-rule",
    "codex-agents",
    "codex-automation",
)


def _host_label(host: AntigravityHost) -> str:
    if host == "cursor":
        return "Cursor"
    if host == "codex":
        return "Codex"
    return "Generic MCP host"


def _workflow_lines() -> list[str]:
    return [
        "1. At the start of every substantial user task, call `mnemos_retrieve` with a focused query tied to the task.",
        "2. Use `current_scope=project`, set `scope_id` to the current repository/workspace name, and include `allowed_scopes=project,global` unless the user asks for broader scope.",
        "3. During execution, call `mnemos_store` only for durable facts:",
        "   - stable user or maintainer preferences",
        "   - project architecture decisions and rationale",
        "   - environment and tooling facts that will matter again",
        "   - recurring bug patterns and their fixes",
        "4. Before finishing substantial work, call `mnemos_consolidate`.",
        "5. If a retrieved memory looks suspicious, call `mnemos_inspect` before storing a correction.",
        "6. Never store secrets, credentials, tokens, or one-off transient chatter.",
        "7. If Mnemos MCP tools are unavailable, continue normally without blocking work.",
    ]


def build_antigravity_policy(host: AntigravityHost) -> str:
    if host not in ANTIGRAVITY_HOST_CHOICES:
        raise ValueError(f"Unsupported host: {host!r}")

    intro = [
        f"Mnemos Antigravity Autopilot Policy ({_host_label(host)})",
        "",
        "Use Mnemos memory tools automatically for substantial coding work.",
    ]
    if host == "codex":
        intro.append(
            "Add this workflow to your repo-level `AGENTS.md` so Codex uses Mnemos via MCP consistently."
        )
    intro.append(
        "This is a soft-auto workflow built on MCP plus host instructions, not a promise of host-level hard auto-capture hooks."
    )
    return "\n".join([*intro, "", *_workflow_lines()]).strip()


def _build_cursor_rule() -> str:
    body = [
        "---",
        "description: Use Mnemos MCP memory automatically for substantial work in this repository.",
        "alwaysApply: true",
        "---",
        "",
        "Use Mnemos through MCP automatically on every substantial coding task in this repository.",
        "",
        *_workflow_lines(),
    ]
    return "\n".join(body).strip() + "\n"


def _build_codex_agents_block() -> str:
    body = [
        "## Mnemos Memory",
        "",
        "Use Mnemos through MCP automatically on every substantial coding task in this repository.",
        "",
        *_workflow_lines(),
    ]
    return "\n".join(body).strip() + "\n"


def _build_codex_automation_prompt() -> str:
    body = [
        "Run a Mnemos workspace hygiene check for this repository.",
        "",
        "1. Verify Codex still has Mnemos MCP configured and that the repo-level `AGENTS.md` still contains the Mnemos memory block.",
        "2. Run `mnemos-cli doctor` and `mnemos-cli stats` in the repository root.",
        "3. Summarize degraded checks, missing setup, or retrieval/storage risks.",
        "4. If `AGENTS.md` is missing the Mnemos block, propose the output of `mnemos-cli antigravity codex --target codex-agents` instead of silently editing unrelated instructions.",
        "5. Do not claim automatic host hooks; this Codex path depends on MCP plus repo instructions.",
    ]
    return "\n".join(body).strip() + "\n"


def build_antigravity_artifact(host: AntigravityHost, target: AntigravityTarget) -> str:
    if host not in ANTIGRAVITY_HOST_CHOICES:
        raise ValueError(f"Unsupported host: {host!r}")
    if target not in ANTIGRAVITY_TARGET_CHOICES:
        raise ValueError(f"Unsupported antigravity target: {target!r}")

    if target == "policy":
        return build_antigravity_policy(host) + "\n"
    if target == "cursor-rule":
        if host != "cursor":
            raise ValueError("The cursor-rule target is only valid for the cursor host.")
        return _build_cursor_rule()
    if target == "codex-agents":
        if host != "codex":
            raise ValueError("The codex-agents target is only valid for the codex host.")
        return _build_codex_agents_block()
    if target == "codex-automation":
        if host != "codex":
            raise ValueError("The codex-automation target is only valid for the codex host.")
        return _build_codex_automation_prompt()
    raise ValueError(f"Unsupported antigravity target: {target!r}")
