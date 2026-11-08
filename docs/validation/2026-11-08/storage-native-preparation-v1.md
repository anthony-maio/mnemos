# Patch native-host preparation

The current draft PR #9 head `fcf02a2e` passed all eight jobs in
[CI run 42503418154](https://github.com/anthony-maio/mnemos/actions/runs/42503418154).
The PR body now records that terminal result. No merge or publication occurred.

Actual Claude Code execution is pending a fresh login to the separate validation
profile. The preflight found fewer than the required 3600 seconds remaining and
stopped before creating the output directory or launching Claude. The user was
asked to run the existing isolated login script. Normal credentials were not
used or refreshed. This blocks this host check, not independent goal work.

Private runner `.venv/check-storage-native-claude-v1.py` prepares two bounded
fresh-process phases against installed patch wheel SHA256
`a3abff76729ea4cc0fa0e8fbba8a087d60e50b8fb5c2324e6222ff44e6ff4463`:
store/inspect one synthetic project fact, then restart and retrieve/inspect the
same durable chunk. Each phase allows 180 seconds and a USD 0.50 host budget.
Explicit direct MCP configuration points at the verified installed package;
the memory backend uses mock/simple providers and a disposable SQLite database.
Builtin host tools are disabled and only the phase's named memory tools are
allowed. Output checks bind actual tool results to exact content, scope and
chunk ID, then independently reopen SQLite read-only.

Credential copies are limited to protected disposable homes and removed during
cleanup. The runner compares normal and validation credential-file hashes before
and after execution. Review identified child-process cleanup, cleanup receipt
and unisolated version-command gaps. The version command was removed. The MCP
bootstrap now records PID plus Windows creation time; cleanup verifies that
identity before stopping a live owned child. Cleanup errors mark failure and
are recorded even when another cleanup step fails.

Three harmless child-process checks pass: a mismatched creation time is not
killed, an owned sleeper is terminated, and its terminal state is recognized.
Syntax compilation passes. Final preflight again reports authentication required,
with no native host launched and no test workspace created. These are harness
checks, not native acceptance. Final review of the revised harness found the
three issues resolved and no remaining blocker for this narrow validation.
The reviewer launched no processes and read no credential files.

Prepared runner SHA256:
`ab15c742352ad354a7e8c487ee7c28225d736b7cdedca75074f94774495cacfd`.
The preparation archive is `.venv/storage-native-preparation-v1.zip`; its manifest
covers the runner, preflight logs, cleanup-check results and terminal CI job
metadata. It contains no credentials or native conversation outputs.
Archive SHA256: `c1b7cac6d5f508b288d2d2fa2438c238cc6ea17983c03f37e736567c7724f45b`.
Every manifest entry was read back and verified.

Even a successful run would establish only the patch's actual Claude direct-MCP
store/inspect/restart-recall path. It would not establish plugin hooks, Claude
Desktop, the separate functional correction/forgetting lifecycle, semantic
quality, complete latency or actual daily-use pilot days. Preserve those gates.
