# Native Codex v7 checkpoint

Codex passed all six explicit-memory phases on installed product
`6d0e5ba69f39d55df8623830cb682e14a5b2d607`, the same artifact that passed
[Claude v5](native-claude-v5-checkpoint.md). Wheel SHA-256:
`ecd11eb7ecbb9cc4f14552d9a7253dead7a8348879c35063411bb850abe06036`.
The [earlier Codex v6 pass](native-codex-checkpoint.md) retains its own product
binding and evidence.

Runner `6c12ff94ddf3bc8c15ec3a1c211f8791cba61916` first passed the
[full regression](full-regression-codex-v7.md). The attempt then used the pinned
Codex CLI 0.164.0-alpha.11.1 and Code Mode executable with `gpt-6-astra`, under
the committed [v7 protocol](../../plans/2026-11-06-native-codex-review-selection-v7.md).
All 57 installed dependency versions matched the verified environment.

| Phase | Result | Correlated MCP calls |
| --- | --- | ---: |
| Teach and observe completed job | Passed | 3 |
| Restart recall and evidence inspection | Passed | 2 |
| Correct the same fact and verify completion | Passed | 6 |
| Restart and recall corrected version | Passed | 2 |
| Review and apply forgetting | Passed | 4 |
| Recall after restart and original-source replay | Passed | 2 |

All 19 calls passed native ID correlation and result checks. The controller
verified the original fact identity through correction, reviewed selection,
deletion from the original SQLite tables, suppression of one exact original
source replay, final absence and retained/unrelated/foreign controls. Each
phase used one prompt in a fresh native home. No failed phase was retried.

SQLite integrity and foreign keys passed. All twelve recorded owned processes
were absent, the service descriptor and copied credentials were removed, and
the normal credential file stayed byte-for-byte unchanged. The
[receipt](native-codex-v7-receipt.json) binds 788 verified private archive files.
Archive SHA-256:
`0e6728c62e0982c3ff50c177f418dae42dad1a50bd3d55da51bca3efb3b3085f`.
Streams, prompts, native sessions and databases remain private.

This is one synthetic run on the named wheel and runtime. It does not validate
the separate pilot launcher or its proposed host patches, automatic capture,
extraction, answer-support quality, native tokenization, reliability across
repeated runs, actual pilot use or publication. No live configuration changed.
