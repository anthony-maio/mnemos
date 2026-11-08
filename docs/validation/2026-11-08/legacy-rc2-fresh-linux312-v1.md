# Fresh Linux installed check for 0.6.1rc2

The unchanged rc2 wheel passed all 90 installed checks on Debian under WSL2 with Python 3.12.13. This includes three automated MCP stdio roundtrips. There were no failures, errors or skips; JUnit duration was 3.143 seconds.

A new environment independently resolved the wheel's MCP extra, pytest and pytest-asyncio. The dependency check passed. Downloads may use the resolver cache. Tests ran outside the source tree with a disposable home, mock LLM and simple embeddings. The source candidate was clean at `a81dc537c82af41feeeb2cd78fd98a9c0f113ef9` before preparation.

The post-run audit checked JUnit counts, installed module origins and all 38 installed package files against the original wheel. Copied tests and external configurations match the Windows check byte for byte. The consumer contains no source package. The checks cover transaction rollback, concurrent source deletion, caller-owned transactions, content/access timestamp separation, dataset packaging, benchmark diagnostic evidence and temporal word boundaries.

- Wheel SHA-256: `8c4ba8464eb1ce66c12e0b1ccaf8bcf31129de3cc322e482b1a672f2f818ec2b`
- Evidence archive SHA-256: `a1f4376220b5e5466cb94e5d49a904f8c8917ec0bcbab937a138f6a06e045cfa`
- JUnit SHA-256: `6ce70b0c19b8e5e52c7a5ae05f4a88b5d70efc3221a916ab44bec4023cc2d5ec`

The [receipt](legacy-rc2-fresh-linux312-v1.json) records the exact platform, dependencies and hashes. Logs, verifier, audit and raw results are retained in `.venv/legacy-rc2-fresh-linux312-v1-evidence.zip`. All invoked processes completed.

This establishes one Linux environment on the same artifact already checked on Windows. It does not establish other Linux distributions or Python versions, macOS, all optional extras, hosted CI or interactive native-host acceptance. The earlier rc1 benchmark remains failed and was not rerun. Quality, complete-command latency, fresh acceptance and the original two-week pilot remain open. Nothing was published or installed into a live host; the optional pilot remains unchanged.
