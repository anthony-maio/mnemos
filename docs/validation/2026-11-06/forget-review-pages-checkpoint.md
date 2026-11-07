# Paged forget-review preparation

Product `5de1844f` adds bounded MCP previews and digest-bound short selections.
The existing storage deletion guards remain in force. Twenty-eight new checks
cover lossless paging, long text, Unicode, changed snapshots, strict argument
types, invalid or mixed refs, required-source selection and an independent
SQLite writer changing a candidate between resolution and mutation.

The product verification covered 77 distinct focused tests: a 75-test run,
then the complete 28-test new file with two additional transport/dependency
checks. Formatting and strict types passed for both affected source modules.
The source and wheel builds, Twine and a fresh [installed smoke check](installed-smoke-5de1844f.json)
passed. The wheel SHA-256 is
`a5907017c4875ca574b13ce9e6c75c1ba65bafd759ff0597bd44bef8a2660ecb`.

The [offline rendering measurement](forget-review-pages-rendering.json) reads
the original 50,711-byte v3 preview without running a model. All 19 candidates
and 22 additional sources reconstruct exactly across five pages. Wrapped page
sizes are 11,115, 11,173, 10,100, 11,062 and 6,490 bytes. This establishes bounded
delivery for that saved response, not a native pass or measured token savings.

Verifier `aebe2a4e` independently reconstructs native pages and resolves short
refs before applying the original inspection, dependency and receipt checks.
Its 77-check focused run includes 36 new page/ref cases. Gaps, overlap, missing
or reordered pages, changed snapshot identities, altered text, malformed version
types and mismatched selections fail. Old single-preview checks still pass.
The verifier changes were developed in a separate worktree while the product
suite ran, then integrated without changing product modules.

The first full product run failed because two historical fixture directories
were absent from the sparse checkout: 2,749 passed, four skipped, two failed
and 19 setup errors. Its [original receipt](full-regression-5de1844f-missing-fixtures.json)
and private archive are retained. All 208 restored files match their Git bytes;
the sparse patterns now include them. Both affected test files pass all 50
checks with no source repair. A fresh full run is required and remains pending.

The isolated native installation exposes the paging/ref parameters and matches
all 57 package versions from the preceding environment. The prospective
[Claude v4 protocol](../../plans/2026-11-06-native-claude-review-pages-v4.md)
binds this wheel, unchanged native prompts and the original phase/time/cost
limits. Inference must wait for the passing full regression result and the
committed protocol/runner binding.

Claude v3 remains failed. No new native attempt, live configuration change,
publication or pilot day is claimed by this preparation checkpoint.
The [preparation receipt](forget-review-pages-preparation.json) binds the source
files, failed and passing focused checks, installation evidence, prospective
runner/protocol and private archive.
