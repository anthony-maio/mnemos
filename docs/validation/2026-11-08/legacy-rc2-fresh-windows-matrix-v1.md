# Fresh Windows dependency matrix for 0.6.1rc2

The unchanged rc2 wheel passed 90 installed checks in each of three newly created Windows environments: Python 3.10.18, 3.11.13 and 3.12.11. There were 270 executions of 90 distinct cases, including nine automated MCP stdio roundtrips, with no failures, errors or skips.

Each environment independently resolved the wheel's MCP extra plus pytest and pytest-asyncio. Dependency checks found no conflicts. Python 3.10 resolved 44 packages; Python 3.11 and 3.12 each resolved 41. This is a fresh environment and dependency solve, not proof that package downloads bypassed the resolver's cache. Exact versions and logs are retained.

The checks cover installed transaction rollback, cross-process source deletion, caller-owned transactions, content/access timestamp separation, the bundled dataset pack, per-query evidence and temporal word boundaries. The same copied test and external configuration bytes were used across all three environments. Mock LLM and simple embeddings keep the automated MCP tests local. No closed ranking comparison was rerun.

An independent audit verified all three JUnit records, package module origins and the 38 installed package files against the original wheel after testing. The consumer directories contained no source package. No live installation, configuration or memory database changed.

| Python | Installed checks | Automated MCP cases included | JUnit duration |
| --- | ---: | ---: | ---: |
| 3.10.18 | 90 | 3 | 23.391 s |
| 3.11.13 | 90 | 3 | 21.799 s |
| 3.12.11 | 90 | 3 | 16.798 s |

Wheel SHA-256 remains `8c4ba8464eb1ce66c12e0b1ccaf8bcf31129de3cc322e482b1a672f2f818ec2b`. Matrix archive SHA-256 is `2bfc6a3f3632d70614c31038dbc9e460035bb68c0996b2b6c103c4c1fb500932`. The [receipt](legacy-rc2-fresh-windows-matrix-v1.json) records exact dependencies and evidence hashes. The separate initial Python 3.12 audit is retained in `legacy-rc2-fresh-dependencies-v1.json` and archive `de2c256009c4eb6681cf5c112ededed4016a5d73ae29b85524a0fbf3f2b14ac1`.

This extends the exact artifact's Windows installed evidence. It does not establish Linux/macOS behavior, all optional extras, hosted CI, interactive native-host acceptance, retrieval quality, complete-command latency or actual pilot use. The earlier rc1 benchmark remains failed. All build/test sessions are terminal. Nothing was published or activated.
