# Installed doctor checkpoint

This archive records one isolated installed-wheel doctor run for the 6600dc2 source and verified wheel. The v5 run passed three configuration-readiness cases: an empty SQLite store reported ready, one legacy unscoped row reported degraded, and a mock cognition configuration reported not_ready. The mock case was a rejection check; the embedding provider state was still required to remain not_checked.

The child processes used a socket guard that blocked the configured loopback Ollama endpoint and non-loopback connections before importing Mnemos, while permitting Windows asyncio loopback plumbing. This is configuration-readiness evidence, not a bare console doctor run and not live provider reachability evidence. No model or provider request was made.

The report contains exact child command arguments, stdout and stderr hashes, full installed Mnemos Python-file hashes against the wheel, and before/after import provenance. Temporary SQLite databases and private TOML configuration files were intentionally excluded from this archive.

Historical report statuses were v1=failed, v2=failed, v3=passed, and v4=passed. These are preserved attempts, not combined results. Prior runner bytes are included only where they existed in the historical output directories.

The evidence index covers every archived file except the index itself, whose self-hash would be circular.
