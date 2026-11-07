# Codex v11: complete explicit-memory lifecycle verified

The current installed artifact passed all six native phases, with 19 correlated
MCP calls, complete advertised tool inventories and one suppressed replay of
the original source. The independent audit verified that every phase received
the revised installed host instructions and the unchanged lifecycle request.

Product `c80eb051` improves the shipped workflow guidance and the MCP preview
definition. The host is told to read each tool's live definition before first
use, inspect the target, and supply its current version when requesting a
forget preview. The preview parameter identifies `fact.version` as its source.
It remains required and strictly typed. Dispatch, storage, stale-state checks,
selection rules and native acceptance criteria did not change.

The change follows [v10's failed attempt](native-codex-v10.md). That trace
showed two missing-version requests before the host supplied the required
argument. The host had read tool names but had not looked up the preview
definition. Its generated instructions explained expected versions for
correction without explaining the preview sequence. V10 remains failed;
the new result is one attempt on a different installed artifact.

Ninety-five focused checks passed across transport validation, stale previews,
scope protection, host integration, CLI instruction generation and inspection.
Two POSIX hook execution checks were skipped on Windows. Missing, boolean,
non-integer and nonpositive preview versions still reject before service
dispatch. Strict type checking passed on both changed modules, and Black passed
on all three changed files. Captured generated artifacts and the actual FastMCP
definition confirm the required arguments and read-only annotation.

The wheel built and passed metadata checks. Its
[installed CLI/MCP smoke](installed-smoke-c80eb051.json) passed real BGE
embedding, evidence inspection, restart recall and foreign-project isolation.
The fresh external native environment matched 95 package files, 84 committed
Python files and 57 dependency versions. All 17 embedding assets retained
their hashes. Its generated Codex instructions matched the reviewed artifact.
Wheel SHA-256:
`edbbbe35220770b800ae4afcc7ac79a305de6f962214ccd905aea3748194247f`.

The [v11 protocol](../../plans/2026-11-07-native-codex-preview-guidance-v11.md)
was committed at `3319da05` before inference. It kept the original six lifecycle
requests, the v9 complete-inventory observation, pinned Codex CLI
0.164.0-alpha.11.1 and Code Mode binaries, `gpt-6-astra`, high reasoning effort
and 180-second phase deadlines. No phase retry or corrective message was used.
The unchanged trace and lifecycle harness retains its 179 focused passes.

| Phase | Calls | Verified behavior |
| --- | ---: | --- |
| Teach | 3 | One explicit assertion; original durable job completed |
| First restart recall | 2 | Same fact ID, version one and retained evidence |
| Correct | 6 | Inspected the current version; same-ID correction job completed |
| Second restart recall | 2 | Same fact ID, version two and corrected evidence |
| Forget | 4 | Current-version preview and reviewed deletion, with no failed call |
| Final restart recall | 2 | Forgotten content absent; retained review convention recalled |

The independent audit reconstructed all 19 calls through native model, code
cell and MCP runtime IDs. Five inspections used `fact_id`. It verified the
exact configured inventory in each phase, the original prompt bytes after
removing the unchanged observation prefix, and delivery of the installed
instruction artifact. Prompt-file hashes and native prompt-string hashes are
recorded separately to account for Windows line endings.

One exact original-source replay was suppressed after deletion. The target and
forgotten labels were absent from the final database and fresh native context.
All three retained, unrelated and foreign controls stayed unchanged. SQLite
integrity and foreign-key checks passed without changing database bytes. The
audit made no model calls.

Cleanup passed. All 13 recorded process identities had exited; runtime
descriptors and copied credentials were removed. Normal authentication,
configuration and model assets were unchanged. The private archive contains
881 verified files, including the wheel, generated instructions, native streams,
database, checks and audit. SHA-256:
`e0f37a4744ad223594b70a0a52a9fcea054c9cd5eb983401615c8dbeda1fd009`.

This establishes one complete synthetic native lifecycle on the named artifact.
It does not establish a reliability rate, qualify experimental extraction or
answer support, validate the resident hook launcher or drafted pilot settings,
meet the failed hook latency target, or count as daily use. The pilot remains
inactive with zero days. The previous full regression is still bound to
`9c84909f`; the current change has focused and installed/native evidence.
Canonical delivery, current full regression, pilot entry checks and actual use
remain open. No live configuration or publication changed.

- [Sanitized receipt](native-codex-v11-receipt.json)
- [Earlier full regression](full-regression-native-hook-v1.md)
