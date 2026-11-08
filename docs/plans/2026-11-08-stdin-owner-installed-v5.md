# Installed stdin-controlled owner regression

Use v5 wheel bbfc09d149959669c0098c01f82dcda409f3566cfebd603dff740a6c16fe29be and the unchanged native v3 launcher. Reuse the v1 owner main-thread stdin command loop and PIPE, with separate JSON and SQLite owners, services and databases. Keep the original 30-second native timeout and normal replay. Three alternating pairs, six identical-per-pair synthetic captures, no sample replacement. This is a bounded reproduction check, not latency acceptance or a new backend performance comparison.

Freeze helper, package and binary hashes before launching. Each capture must return a queued receipt, reach one event and one completed job, and leave both queue formats empty. Reopen both stores and require three events and completed jobs per arm. Verify recorded owner and launcher processes absent and ports closed, including an owner that exits early. Preserve any failure and earlier studies. Do not change live configuration or memory databases.
