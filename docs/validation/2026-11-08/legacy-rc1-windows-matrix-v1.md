# Installed prerelease checks across Windows Python versions

The exact 0.6.1rc1 wheel from c36ca712 passes the same installed storage and
MCP check set on Windows Python 3.10.18, 3.11.13 and 3.12.11. The Python 3.11
result is retained from the original artifact check; this increment adds the
3.10 and 3.12 runs without rebuilding or changing the wheel.

| Python | Storage and recency | MCP roundtrips | Package files verified |
| --- | ---: | ---: | ---: |
| 3.10.18 | 39 passed | 3 passed | 35 |
| 3.11.13 | 39 passed | 3 passed | 35 |
| 3.12.11 | 39 passed | 3 passed | 35 |

The 126 executions represent 42 distinct cases, not 126 independent behaviors.
An independent XML audit verifies identical case IDs and copied test bytes in
all three cells, with no failures, errors or skips. The MCP command selects
three roundtrips and deselects one unrelated static configuration test.

Each new cell installs the unchanged wheel into its own external target with
no dependency installation. Existing isolated legacy environments provide the
dependencies, whose actual versions are retained. All 35 pre-existing package
files remain unchanged in each environment. Runtime version and installed
metadata agree; imported package paths point to the new target. Tests use
fresh databases, copied manifests and wrapper, and a disposable home.

Coverage includes another process deleting a source, graph rollback and caller
transactions, pending full-text edits, content/access separation, legacy schema
migration, concurrent opening of an existing WAL database, and generic/Claude
sample MCP roundtrips. The latter use mock LLM and simple embeddings. They are
automated configuration checks, not interactive native-host sessions.

The [receipt](legacy-rc1-windows-matrix-v1-receipt.json) identifies the wheel,
three environments, case counts and private evidence archive. It retains the
original Python 3.11 records plus new raw results, JUnit records, origins,
dependency metadata and helpers. All new commands exited zero. The artifact,
source checkout, normal host settings and inactive pilot remain unchanged.

This is a Windows installed-check matrix. It does not repeat the complete
353-case source suite, resolve fresh dependencies, establish Linux or hosted CI
coverage, or override the original benchmark failure. Semantic quality, complete
host latency, interactive acceptance, actual daily-use pilot and publication
remain open. Schema-3 upgrade and rollback limits still apply.
