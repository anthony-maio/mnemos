# Install

Use this file when the user needs the concrete Codex setup path.

## 1. Install the local Codex skill

Copy this folder to your Codex skills directory if you want Codex to have local Mnemos guidance:

- macOS/Linux: `~/.codex/skills/mnemos-codex/`
- Windows: `%USERPROFILE%\.codex\skills\mnemos-codex\`

If you are working from this repo, copy the checked-in `skills/mnemos-codex` directory as-is.

## 2. Install Mnemos with MCP support

```bash
pip install "mnemos-memory[mcp]"
mnemos ui
```

Prefer the control plane. It writes the canonical `mnemos.toml`, previews/applys Codex host setup, and runs smoke checks.

## 3. Wire Codex to Mnemos

The Codex path is MCP-first. The control plane should write `~/.codex/config.toml` and the repo `AGENTS.md` Mnemos block for you.

Manual Codex MCP shape:

```toml
[mcp_servers.mnemos]
command = "mnemos-mcp"

[mcp_servers.mnemos.env]
MNEMOS_CONFIG_PATH = "~/.config/Mnemos/mnemos.toml"
```

Generate the repo memory policy block with:

```bash
mnemos-cli antigravity codex --target codex-agents
```

Optional maintenance prompt for Codex Automations:

```bash
mnemos-cli antigravity codex --target codex-automation
```

## 4. Validate

Run:

```bash
mnemos-cli doctor
```

Then verify one real loop:

1. recall: `mnemos_retrieve`
2. normal coding work
3. curator: `mnemos_store`
4. finish: `mnemos_consolidate`

If you are validating Codex specifically, make sure the query uses the repo scope:

```text
current_scope=project
scope_id=<workspace-or-repo-name>
allowed_scopes=project,global
```
