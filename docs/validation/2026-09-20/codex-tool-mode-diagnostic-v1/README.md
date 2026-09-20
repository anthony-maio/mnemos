# Codex model catalog and feature diagnostic

The failed [three-host v2 attempt](../three-host-native-v2/README.md) used
`gpt-5.6-sol` and model-authored `exec` calls, despite its requested
`--disable code_mode_host` setting. This read-only diagnostic queried the
installed 0.146.0 executable with that flag and the same disposable Codex
home. All four `code_mode*` feature flags reported false.

The saved model catalog from that session independently marks `gpt-5.6-sol`,
`gpt-5.6-terra`, and `gpt-5.6-luna` as `tool_mode: code_mode_only`. The visible
`gpt-5.5` entry has a null `tool_mode`. This is evidence that the model catalog
and feature flags represent separate settings; it is consistent with the
observed wrapper path. It does not establish the CLI's internal precedence
rules or prove that a null value yields direct calls.

The next prospective profile explicitly selects `gpt-5.5`, verifies the
actual model from every native turn-context record, and retains strict
model-to-MCP ID correlation. No model request, Mnemos service call, live
configuration edit, or credential read was performed for this diagnostic.

`selected-catalog.json` preserves selected catalog fields and binds the
original cache bytes by hash, size, timestamp, and path. Full model
instructions and unrelated metadata are omitted. `features.stdout.txt` and
`features.stderr.txt` preserve the actual feature-query output. The index
binds those four files, including the archive attributes; this README and
the index itself are excluded.
