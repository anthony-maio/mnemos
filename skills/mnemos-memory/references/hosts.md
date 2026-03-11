# Host Setup

Use this file when the user needs host-specific setup details.

## Common install

```bash
pip install "mnemos-memory[mcp]"
mnemos ui
```

Prefer the control plane. It writes canonical config, previews host integration, and runs smoke checks.

## Claude Code / Claude Desktop

- Best path: use the control plane or the bundled plugin packaging.
- Plugin-first install path:

```text
/plugin marketplace add anthony-maio/mnemos
/plugin install mnemos-memory@mnemos-marketplace
```

- Manual MCP shape:

```json
{
  "mcpServers": {
    "mnemos": {
      "command": "mnemos-mcp",
      "env": {
        "MNEMOS_CONFIG_PATH": "~/.config/Mnemos/mnemos.toml"
      }
    }
  }
}
```

## Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "mnemos": {
      "command": "mnemos-mcp",
      "env": {
        "MNEMOS_CONFIG_PATH": "~/.config/Mnemos/mnemos.toml"
      }
    }
  }
}
```

Cursor is currently documented/best-effort, not part of the release-blocking validation set.

## Codex

- Mnemos for Codex is MCP-first, not plugin-first.
- Register `mnemos-mcp` in `~/.codex/config.toml`.
- Add repo policy text so the agent actually uses the tools consistently.

Minimal config:

```toml
[mcp_servers.mnemos]
command = "mnemos-mcp"

[mcp_servers.mnemos.env]
MNEMOS_CONFIG_PATH = "~/.config/Mnemos/mnemos.toml"
```

Policy generator:

```bash
mnemos-cli antigravity codex
```

## Generic MCP hosts

If the host can run a stdio MCP server, point it at `mnemos-mcp` and pass `MNEMOS_CONFIG_PATH` to the canonical config file.
