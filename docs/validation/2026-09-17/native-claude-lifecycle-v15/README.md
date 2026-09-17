# Native Claude lifecycle v15 evidence

This archive preserves the failed v15 native lifecycle, its adapter timing prerequisite, and one bounded installed-package MCP catalog control. The v15 Claude process completed one turn, but its `system/init` row reported `tools=[]` and `plugin:mnemos-memory:mnemos` as `pending`. The parser therefore found no allowlisted operation calls or results and correctly refused phase 1. Phase 2 was not run.

The direct installed-package MCP exchange used the v15 private selectors and the pinned 56182d9 wheel environment. It returned successful `initialize` and `tools/list` responses in 1.853 seconds and exposed all 14 Mnemos tools. This shows the wrapper and MCP server catalog work in isolation. It does not prove Claude native startup readiness. The v12 native control reached `connected` and executed correlated health and retrieve calls.

The next diagnostic concerns Claude-side MCP discovery timing or readiness under the v15 invocation. Current Claude documentation describes [background connection and startup waits](https://code.claude.com/docs/en/env-vars) and [loading a server's tools at startup](https://code.claude.com/docs/en/mcp). Those documents do not establish the internal cause in the installed Claude 2.1.191 run. No native rerun was performed here.

The direct MCP control was executed through an inline Python command. Its report records the interpreter, wrapper, private selectors, request methods, response summaries, and stream hashes; no standalone helper or raw stream is included for that control. This limits independent reproduction of the exact diagnostic invocation.

The evidence preserves no raw native stream in this archive. Raw v15 files remain in the private fixture referenced by the reports. There is no model-quality or full lifecycle claim in this archive.

Files are byte-hashed in `evidence-index.json`. Runner copies use `.py.txt` so archived evidence is not discovered as active source by formatter or type checker.
