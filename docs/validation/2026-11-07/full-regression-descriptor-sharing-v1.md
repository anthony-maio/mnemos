# Full regression after the Windows descriptor fix

The full suite passes on source `0456cb5d8e6622050a5c139c2722e691b6b93f2a`, with product
code from `3d1866d50d6df4d46666155006792ad46a9c6f11`: 3,560 passes, no failures or errors,
and four platform skips across 3,564 cases. Pytest exited 0 and reported
555.60 seconds.

This includes the competing-launcher test that failed in the
[preceding run](full-regression-preview-guidance-v1.md), plus all 12 new
descriptor cases. The earlier failed result and unsuccessful first repair
attempt remain archived. This run used a fresh output directory and test root.

The audit verified unchanged tracked source, the explicit canonical website
and all 208 restored historical fixtures. Both recorded process identities
had exited. The skips are the same two WSL Hermes checks requiring an external
interpreter and two hook-command checks requiring a POSIX executable and shell.

The [installed artifact](descriptor-sharing-install-v1-receipt.json) matches
96 package files and 85 committed Python files. Its CLI/MCP restart, recall,
inspection and project isolation checks passed with real BGE embeddings.
All 57 dependency versions matched and all 17 cached model assets stayed
unchanged. The earlier [native Codex v11 result](native-codex-v11.md) remains
bound to c80eb051; this regression did not rerun a native model.

The [full-suite receipt](full-regression-descriptor-sharing-v1-receipt.json)
records the source, counts, skips and artifact binding. Its private archive
contains 10 members and 83,155 bytes,
with SHA-256 `0cbbe626644a181061c4ec22de14962b2208510249f8cadf945ea86efc2042cd`. Each archived member was checked
against its manifest hash after creation.

After the README and reports were updated, all 12 documentation consistency
checks passed against the explicitly selected canonical website. No product
code changed after the full run.

The [updated delivery bundle](delivery-review-v2.md) prepares this exact
tested tree against the preserved delivery base. It remains unapplied.
Quality and complete hook latency thresholds, canonical integration, website
alignment and actual daily use remain open. The private pilot is inactive
with zero days. No live configuration or published artifact changed.
