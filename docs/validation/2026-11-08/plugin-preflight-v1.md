# Claude plugin preflight

PR #9 source `fcf02a2e74a5fdf7595aadf43676e2c3ae124e84` was archived into
disposable private directories. Each extracted file was checked against the
archive before validation. The source checkout was clean and remained unchanged.
No authentication was copied and no model was requested.

Claude's directory-level `plugin validate --strict` selected the marketplace
manifest, not the plugin manifest. That check passed. Explicitly targeting
`.claude-plugin/plugin.json` produced one warning: root CLAUDE.md is not loaded
as plugin context. Strict validation failed on that warning. Ordinary validation
passed with the same warning. The file describes repository development guidance;
neither result establishes that its instructions reach plugin consumers.

Private receipts are retained in `.venv/plugin-manifest-v1`, `-v2` and `-v3`.
Each has source/CLI hashes, stdout/stderr and process accounting. These are
manifest checks only. Actual plugin MCP calls, automatic hook effects, managed
installation and Claude Desktop remain unverified by this increment.

## Owned process cleanup

The reusable Windows helper assigns a waiting broker to a new job before
delivering the target command. Job cleanup covers descendants even after a
successful parent exit. It does not operate on process names or rediscovered
PIDs. This is test-process containment, not a security sandbox. It requires the
optional pywin32 validation dependency; product dependencies are unchanged.

Review identified a synchronous input-write timeout gap. A regression confirmed
that Windows communicate(input, timeout=...) still blocked for roughly 60
seconds when the broker never read a 1 MB payload. The helper now delivers input
in a writer thread, shares one deadline across delivery and execution, terminates
the job on failure, then joins the writer before closing its stream.

Final targeted validation: five tests passed in 5.39 seconds, covering normal
exit, timeout with child/grandchild, assignment failure before command delivery,
surviving descendants after parent exit, and blocked input delivery. The final
JUnit is `.venv/windows-owned-job-v5.xml`; earlier failed receipts are preserved.
Read-only review found no remaining concrete blocker after the writer-thread
repair. All three manifest runs recorded zero active job processes after cleanup.

This does not close native-host acceptance, semantic quality, complete latency,
daily-use pilot or publication requirements. The next host check should exercise
the unchanged plugin launcher and hook configuration against an isolated database
and the exact installed patch artifact, with durable results verified directly.
