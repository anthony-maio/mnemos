# Storage transaction repair opened for review

[Draft PR #8](https://github.com/anthony-maio/mnemos/pull/8) now exposes the prepared four-file transaction repair for review. Its head is `e02c395137d65fd87a73c7697ecd20aa5a698ecc` on `wt/fix-graph-transactions-pr7`; its base is PR #7's `wt/fix-mcp-v1-ci` at `9c9ebf8406856057aac051862cb6f7bf55a802d8`. GitHub readback confirms draft=true, merged=false, one commit and four changed files. No merge or release occurred.

Fresh remote checks confirm PR #5 is merged, PR #6 and PR #7 remain open, and public master remains `866444378e5a42598bba0548f6460d07adcded10`. The draft is stacked on #7 because the transaction repair and FTS update patch overlap in the storage method. It should be retargeted and rechecked after #7 is merged, not merged into the prerequisite branch merely because GitHub reports no textual conflict.

On the exact draft head, 13 graph-write, graph/FTS and SQLite checks passed on Windows Python 3.12.11. Black passed the three changed Python files. The worktree remained clean. Independent source review confirmed the merge base, four-file scope, write-lock ordering, savepoint rollback and caller commit/rollback tests.

The guarantee covers statement-level insertion failure. Database-wide rollback or lost connections may invalidate the savepoint and are not covered by the tested rollback guarantee. The PR body states this limit.

The earlier full-suite result on this head remains 319/320 on Windows Python 3.11, with a plugin-launch failure. Typing and broader release checks are not claimed green. The separate combined candidate's passing package/full-suite results do not transfer to this draft. The workflow lookup returned no PR-triggered run for this head at the time of the check; no hosted CI result is claimed.

This advances affected-user delivery from local preparation to an open review draft. Approval, prerequisite integration, final validation and release delivery remain outstanding. The broader semantic, latency and pilot gates remain unchanged.
