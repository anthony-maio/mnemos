# Source and installed artifact checkpoint

Source `749f1213381848a0d6040bf8a241829b18b655a1` passed the full suite:
3,221 tests passed and four were skipped, with no failures or errors among
3,225 collected tests. Pytest reported 528.54 seconds. Two skips require an
installed WSL Hermes interpreter; two require a POSIX executable and shell.
HEAD, tracked files, the explicitly selected canonical website and all 208
historical fixture hashes remained unchanged during the run.

The same source built wheel and source artifacts, and both passed Twine checks.
The installed wheel passed its check outside the checkout using real local
BGE-small embeddings: explicit CLI remember, service restart, same-ID recall,
MCP retrieval and source inspection, and exclusion of a foreign project. The
service stopped and its temporary consumer directory was removed. Loading the
Hermes entry point is installation evidence, not a native Hermes session.

An independent audit matched all 90 package files in a second isolated
installation to the wheel. Eighty-three source files also matched across the
wheel, source archive and checkout. All 57 dependency versions matched the
previous validation environment. The installed MCP catalog contains 18 tools
and exposes `response_format="compact"`; `full` remains the default. All five
experimental module defaults remain off.

| Artifact | SHA-256 |
| --- | --- |
| Wheel, version 0.7.0b1 | `2884c412f10ce130aed0afad0910c461ba945657f465f73ec604def78ab3ba2e` |
| Source archive | `b839ee90bf3f716696a6a966298a18f905530b51a4885d3c5eec8e2d696aa503` |

The [receipt](checkpoint-749f1213-receipt.json) independently matches the JUnit
counts and binds all 22 verified private archive members. Archive SHA-256:
`5b8a36308bd432deed1376df8c6e6fedaf27726b577610367268dbb14d57fe4c`.
The JUnit suite duration excludes some pytest startup/reporting overhead.
Both controller sessions exited zero; their recorded processes are absent.

These checks do not establish fresh semantic quality or native-host acceptance
of this new wheel. The [inactive pilot](pilot-entry-audit-v1.md) still uses
`6d0e5ba6`, which retains its own native and recovery evidence. It was not
upgraded or activated. No live configuration, public package or website changed.
