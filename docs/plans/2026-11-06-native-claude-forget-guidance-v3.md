# Native Claude forget-guidance checkpoint v3

Run one new six-phase attempt against product
`5de8b1f1624f3d9cf5b54d284416bb30c884221e`. The wheel is
`mnemos_memory-0.7.0b1-py3-none-any.whl`, SHA-256
`7781fcc3b38f68ad7a6171074701dd1e539eecc9820178343efec93b047819e1`.
The [installed smoke receipt](../validation/2026-11-06/installed-smoke-5de8b1f1.json)
binds this artifact. This binding supersedes v2's product and wheel.

The v2 forgetting attempt included two automatically covered target sources
in the additional-source selection. The tool rejected that first call without
committing. Claude then removed those IDs and committed, but the strict native
gate correctly rejected the attempt. The final restart/replay phase was not
run. Preserve that failure and its [diagnosis](../validation/2026-11-06/native-claude-v2-diagnosis.md).

The new product explains the two source lists in the MCP tool descriptions and
selection parameter schema. Additional source IDs must come from the preview's
top-level `source_events` entries with `selection_allowed=true`; the target's
linked sources are covered automatically. Storage guards and acceptance rules
remain unchanged. The artifact also contains the later experimental support
code, which remains outside production retrieval.

Keep the v2 native prompts unchanged, including the forget prompt. Keep Claude
Code 2.1.191, `claude-sonnet-4-6`, the same synthetic scenario, six fresh native
sessions, 180 seconds and $0.50 per phase, and a $3.00 total ceiling. Make one
attempt with no retries of failed phases. Require every native tool result to
pass the existing strict parser and all existing lifecycle/control assertions.

Commit this amendment, the smoke receipt and the runner's amendment binding
before inference. Use a fresh external installed environment and output root.
Verify that the installed tool catalog contains the clarified schema. Reuse
only the isolated validation credential source after checking at least one hour
of access lifetime. Preserve source credential bytes and delete phase copies.

Retain native prompts, streams, process receipts and failures privately. Verify
model/session/tool provenance, same-ID correction, final restart, exact-source
replay suppression, retained/foreign controls, SQLite integrity and process,
descriptor and credential cleanup. Publish only sanitized findings and hashes.
This synthetic checkpoint contributes no days to the private daily-use pilot.
