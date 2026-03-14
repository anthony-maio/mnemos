# Operations

Use this file when the user asks how Mnemos should work day to day in Codex.

## Hard auto-capture reality check

- Codex can use Mnemos well today through MCP plus strong repo instructions.
- Codex does not get the shipped Claude Code hook layer by default.
- Hard auto-capture is still host-dependent in Codex.
- Codex Automations are useful for maintenance, not for prompt/tool lifecycle capture.

Do not hide this gap. If the user asks whether Codex has Claude-style automatic capture today, the answer is no.

## Recommended operating loop

1. Start substantial tasks with `mnemos_retrieve`.
2. Use `current_scope=project` and set `scope_id` to the repo or workspace name.
3. Store only durable facts with `mnemos_store`.
4. Finish substantial work with `mnemos_consolidate`.
5. Use `mnemos_inspect` if a retrieved memory looks wrong.

## Troubleshooting

Start with:

```bash
mnemos-cli doctor
mnemos-cli stats
```

Check:

- `~/.codex/config.toml` has a `mnemos` MCP entry
- the repo `AGENTS.md` includes the Mnemos block
- `MNEMOS_CONFIG_PATH` points at the expected shared config
- the configured store/provider are actually reachable

If setup drifted, prefer reapplying the control-plane Codex integration instead of editing multiple files by hand.
