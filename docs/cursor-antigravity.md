# Cursor Antigravity (Auto Memory)

Use this guide to make Cursor call Mnemos memory tools automatically without manual prompting each time.

## 1) Register Mnemos MCP in Cursor

Copy [docs/mcp-configs/cursor.json](docs/mcp-configs/cursor.json) into your project `.cursor/mcp.json` (or merge it into your existing config).

Default profile in that config is zero-friction:

- `MNEMOS_LLM_PROVIDER=mock`
- `MNEMOS_EMBEDDING_PROVIDER=simple`
- `MNEMOS_STORE_TYPE=sqlite`
- `MNEMOS_SQLITE_PATH=.mnemos/memory.db`

Upgrade later to Ollama/OpenAI/OpenClaw for higher retrieval quality.

## 2) Generate Antigravity policy text

Generate a host policy prompt with the CLI:

```bash
mnemos-cli antigravity cursor --write .cursor/mnemos-antigravity.txt
```

Then paste that text into your Cursor project/system instructions so the assistant follows it by default.

## 3) What the policy enforces

The generated policy makes the assistant:

- call `mnemos_retrieve` at task start
- call `mnemos_store` for durable facts during work
- call `mnemos_consolidate` before finishing substantial tasks
- use `project` + `scope_id` for cross-project-safe retrieval
- avoid storing secrets and transient chatter

## 4) Live smoke test

In Cursor chat, run this sequence:

1. "Remember that this repo deploys to AWS ECS."
2. "Remember globally that I prefer concise summaries."
3. "What do you remember about deployment in this repo?"

Expected:
- deployment response should include repo-scoped memory
- global preference may also appear
- no manual tool call wording required in your prompt
