# Claude v4: complete pages, incomplete deletion

The new native attempt failed. Teaching, restarted recall, same-ID correction
and corrected recall passed. Forgetting failed, and final restart/replay was
not attempted. The failed result and its original streams remain preserved.

Product `5de1844fa82ccaa9d64324c89aebd96a254526c3` supplied the installed wheel,
SHA-256 `a5907017c4875ca574b13ce9e6c75c1ba65bafd759ff0597bd44bef8a2660ecb`.
Runner `7f5aec4b587599ccf0592af236cc5b8d0fc2a5eb` kept the frozen prompts,
scenario, Claude Code 2.1.191, `claude-sonnet-4-6`, 180-second and $0.50 phase
limits. The [full regression](full-regression-review-pages-v2.md) passed before
inference. There was one fresh attempt and no failed-phase retry.

The first four phases contain 12 parsed, correlated MCP calls. The forget
stream contains 15 further tool results. Its second request was
`mnemos_reviews`, outside the four permitted tools. Claude denied permission,
and the strict parser rejected the unsupported tool name. The native startup
record advertised all 18 Mnemos tools, including that denied tool. The runner's
permission list did not restrict the advertised catalog. No permission was
granted to that request.

Read-only inspection of the saved trace establishes narrower component results.
All five review pages arrived inline. Their sizes were 11,745, 10,909, 11,859,
11,982 and 3,148 bytes. Independent reconstruction recovered all 18 candidate
facts and 22 additional sources with the original snapshot digest. There was
no persisted-output notice. These observations support the bounded-page repair
in this run; they do not make the failed native lifecycle pass.

Claude inspected and selected six additional facts using `f1` through `f6`,
plus their six required source references. One apply committed, and the saved
selection passes the separate preview, inspection, dependency and receipt
checks. Its coverage reported zero retained exact copies, twelve retained
advisory facts and sixteen retained eligible sources. Zero exact copies did
not establish complete deletion of the requested information.

The final database still contains the old label in an assistant summary and
its evidence. That summary appears as `f7` in the reviewed preview; its quoted
sentence omits the original final period and is classified as a similarity
candidate. Its required source `s9` was also omitted. Two deferred user-source
captures, `s2` and `s5`, contain the old and corrected labels, respectively.
Both were visible, eligible exact-assertion sources with no derived facts.
All three remaining sources predate the preview. This is an incomplete
selection, not evidence that paging hid those records or a later writer added
them. No new replay was run to repair or reinterpret this failure.

The [diagnosis](native-claude-v4-diagnosis.json) binds the original request/result
lines and final database scan. One fact row, three source rows and one evidence
row still contain a forgotten label. Revisions and job payloads contain none.
The [native receipt](native-claude-v4-receipt.json) binds 140 verified private
archive members, SHA-256
`169e3bf7be2f7bb21f86fd2ad0f50b4b88b8b92102bca870d827551ff6f4af27`.
The [diagnostic receipt](native-claude-v4-diagnosis-receipt.json) separately binds
the later read-only analysis. Total reported native cost was $0.6378249.

Cleanup passed. All owned processes are absent, SQLite integrity and foreign-key
checks pass, copied credentials and the service descriptor are removed, and
both source credential files retain their original bytes. No live configuration
changed. The selected private pilot remains unstarted.

Next work should distinguish extraction-review tools from forget review, verify
the native advertised tool boundary, and make review guidance cover advisory
paraphrases and eligible source-only captures. Keep explicit selection,
inspection, source dependencies and atomic deletion guards. Changing the parser
to accept this run would leave the demonstrated deletion failure unresolved.
