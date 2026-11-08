# Patch CI and host-acceptance claim correction

[CI run 42503225584](https://github.com/anthony-maio/mnemos/actions/runs/42503225584)
completed successfully on runtime commit `1319e1c6`. All eight jobs passed:

- Python 3.10, 3.11 and 3.12 each report 329 passing tests.
- Black accepts 66 files and mypy accepts 31 source files.
- Both legacy benchmark repetitions report production_replacement.passed true.
- Tier 1 transport checks report four passes; Tier 2 reports one.
- The release_readiness job reports 12 passing checks and checks that the release
  checklist file exists. It does not execute every item in that checklist.

The legacy benchmark does not satisfy the original fresh semantic evaluation or
complete capture/recall latency gates. Test names mentioning Claude clients do
not prove those applications were used. Inspection of test_mcp_compatibility.py
shows that the tests load configuration and launch its process through the
generic Python MCP client, then exercise store, inspect, retrieve, feedback and
health. They do not launch Claude Code or Claude Desktop.

Documentation commit `fcf02a2e74a5fdf7595aadf43676e2c3ae124e84` changes only
CHANGELOG.md, docs/client-compatibility-matrix.md and docs/release-checklist.md.
The matrix now distinguishes automated plugin/config transport passes from
pending native acceptance for the patch candidate. Tier 1 obligations remain
unchanged. The checklist names mnemos.storage_failure separately and states its
coverage limits. The changelog includes the new diagnostic and claim correction.
Eight documentation consistency tests and git diff --check pass.

The change is pushed to [draft PR #9](https://github.com/anthony-maio/mnemos/pull/9).
Its new [CI run 42503418154](https://github.com/anthony-maio/mnemos/actions/runs/42503418154)
was confirmed queued. Do not transfer the earlier passing status to this head.
Runtime/package checks remain bound to 1319e1c6. Runtime files are unchanged,
but the existing sdist lacks the later documentation corrections.

Private CI archive `.venv/storage-pr9-ci-evidence-v1.zip` contains eight decoded
job logs, terminal run/job metadata and a verified SHA256 manifest. Every entry
was read back and verified. Archive SHA256:
`253274468fed73fbd66ca574d734bf2c3a76545340e5123426959dde190649cb`.
Documentation-test JUnit is `.venv/storage-claims-docs-v1.xml` outside that archive.

Next substantive acceptance work is an isolated actual-host check against the
installed patch runtime, preserving normal host configuration and databases.
The existing functional-memory native runner targets a different API/version;
it must not be relabeled as patch acceptance. Original semantic quality,
complete latency, native lifecycle and pilot obligations remain open. No merge,
release tag, publication or live configuration change occurred here.
