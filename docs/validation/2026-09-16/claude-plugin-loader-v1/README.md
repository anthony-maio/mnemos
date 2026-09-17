# Claude plugin loader metadata

Claude Code 2.1.191 loaded the plugin metadata from clean frozen source `9aba209` through `--plugin-dir`. The list command reported the enabled session plugin `mnemos-memory@inline`, version 0.7.0, and its MCP wrapper. The details command reported SessionStart, UserPromptSubmit, Stop, PreCompact, and PostCompact hooks. The parent checked those values in the preserved outputs as well as their hashes.

The two commands disagree about MCP inventory: list contains `mnemos`, while details reports zero MCP servers. Neither command connects MCP or executes hooks. A native session is still required to verify those paths. No model turn or native memory lifecycle ran, and no installed-wheel acceptance is inferred from plugin metadata. The full 9aba source run separately found a benchmark compatibility regression.

The probe used a disposable Claude config directory and project, copied no auth, and retained normal permission behavior. Before/after source hashes match. Raw records and the executed helper are preserved unchanged. The helper uses a fixed output directory and must not be rerun over this evidence; use a new path/version for any future attempt.
