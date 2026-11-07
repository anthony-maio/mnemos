# Native Claude checkpoint v5

Claude v5 passed the complete explicit-memory lifecycle on installed product
`6d0e5ba69f39d55df8623830cb682e14a5b2d607`, using runner
`15229fb75c0520966e9a7124fcb6efc4a9d9c865`. This was one fresh attempt under the
[committed protocol](../../plans/2026-11-06-native-claude-review-selection-v5.md),
after the [full regression](full-regression-review-selection-v5.md) passed.
The [receipt](native-claude-v5-receipt.json) binds every result to its artifact.

The wheel SHA-256 is
`ecd11eb7ecbb9cc4f14552d9a7253dead7a8348879c35063411bb850abe06036`.
The isolated installation passed its CLI/MCP smoke check and retained the
preceding native environment's 57 package versions. The native host was Claude
Code 2.1.191 with `claude-sonnet-4-6`; all six sessions were fresh.

| Phase | Result | Correlated MCP calls |
| --- | --- | ---: |
| Teach and observe completion | Passed | 3 |
| Recall after restart | Passed | 2 |
| Inspect and correct the same fact | Passed | 5 |
| Recall version two | Passed | 2 |
| Review and forget requested copies | Passed | 15 |
| Recall after deletion, replay and restart | Passed | 2 |

Every startup advertised exactly its allowed tool set. All 29 calls passed
strict native correlation and successful-result checks. Forgetting passed
independent page reconstruction, explicit candidate inspection, source
dependency and exact receipt checks. The requested labels were absent from
the original SQLite tables. Ten exact original-source replays were suppressed
at their original identities. Final native recall preserved the review
convention, excluded the forgotten labels, and retained unrelated and foreign
scope controls.

The host reported $0.7566741 across all phases. Forgetting used $0.4962732 of
its $0.50 phase allowance. This pass establishes neither a comfortable cost
margin nor a reliability rate. The earlier failed native attempts remain
failed; no retry, parser relaxation or revised selection rule reclassified them.

All owned processes stopped, the service descriptor and copied credentials
were removed, and both source credential files remained byte-for-byte
unchanged. SQLite integrity and foreign-key checks passed. The private archive
contains 158 verified members and has SHA-256
`428a31361a462d065a6f55162817b0c0c8e582af3528022c115a89ce47f113c7`.
Raw streams, prompts and databases remain private.

No live host configuration changed. This run does not establish automatic
extraction quality, improved retrieval, current Hermes acceptance, native host
tokenization, completed pilot days or publication. The earlier Codex pass
remains bound to its own tested artifact.
