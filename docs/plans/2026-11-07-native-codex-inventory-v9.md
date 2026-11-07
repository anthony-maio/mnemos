# Native Codex inventory observation v9

Run one fresh six-phase attempt on the unchanged product `cddf8adc` and wheel
SHA-256 `c6f6ff3ddf260a2a736867f01ce923879d68e2dd91ff44abbf0f7a22e1e6982f`.
Keep the v8 lifecycle result and its incomplete catalog check unchanged.

The v8 traces include initial empty discovery results followed by successful
discovery. Some later queries cover the Mnemos namespace, but other phases
still filter by operation. They do not establish the full inventory in every
phase. This amendment adds an explicit observation before each original
lifecycle prompt; it changes no fact, expected answer, permission or score.

Use `--require-tool-inventory`. The first Code Mode cell must contain exactly:

```javascript
text(ALL_TOOLS.map(tool => tool.name).sort());
```

If no `mcp__mnemos__` name is present, permit one identical inventory cell while
startup finishes. The second inventory must contain the exact configured
Mnemos allowlist. A nonempty partial inventory, an extra tool, a third readiness
attempt, a filtered query or a constructed array fails the check. Inventory
cells must finish without overlap and before the first memory operation.

Require the original model-call, code-cell and MCP runtime correlation chain.
Bind both the initial and final inventory responses to the same completed cell,
require one successful text result with sorted unique tool names, compare both
responses and retain their hashes. Recheck all consumed trace bytes. A native
turn or inventory failure ends this attempt; no corrective message or phase
retry is permitted. The one bounded inventory readiness observation is not a
retry of the lifecycle phase.

Preserve the v8 six lifecycle requests byte for byte after removing the new
fixed observation prefix. Record their original hashes and verify parity before
inference and during the independent audit. Keep the pinned Codex and Code Mode
binaries, `gpt-6-astra`, high reasoning effort and 180-second phase deadlines.
Use the verified external cddf8adc installation only after rechecking all 95
package files, 57 dependency versions and 17 embedding assets. All native homes,
projects, registry, memory database and runtime state must be fresh.

The product remains covered by the 3,495-pass source regression checkpoint.
Run focused tests for the changed trace parser, native process boundaries,
explicit lifecycle and reviewed forgetting before committing this amendment
and harness. No product code or package rebuild is needed for this measurement.

Retain all original job-completion, evidence, same-ID correction, reviewed
deletion, exact source replay, retained/foreign control and final restart gates.
Keep automatic enrichment and biomimetic modules disabled. Preserve normal
host configuration and credential bytes; require at least one hour on the
login, protect and remove every copy, stop owned processes and verify cleanup.
Keep raw model streams private and report only sanitized results.

This remains a synthetic native lifecycle. It does not accept the opt-in
resident hook launcher, drafted pilot configuration, answer-support quality,
failed hook latency target or actual daily use.
