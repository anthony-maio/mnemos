# Installed 0.6.1 patch candidate with storage diagnostics

Source: `1319e1c62ca21d46f5f38e57dd485f39c0d6f46f`, clean worktree
`mnemos-storage-release-merged`. [Draft PR #9](https://github.com/anthony-maio/mnemos/pull/9)
targets master `588a190b`. It is not merged, tagged or published.

Built an sdist and then a wheel from that sdist using isolated Hatchling 1.33.0.
Both pass Twine checks. Artifacts remain private:

| Artifact | SHA256 |
| --- | --- |
| mnemos_memory-0.6.1-py3-none-any.whl | a3abff76729ea4cc0fa0e8fbba8a087d60e50b8fb5c2324e6222ff44e6ff4463 |
| mnemos_memory-0.6.1.tar.gz | 247f2037aa1fca8a83bd1255a708196c9c4561a91498062131155029b41286bc |

All 35 package files match source, sdist, wheel and installed bytes; wheel RECORD
hashes and sizes verify. The isolated consumer is outside the checkout at
`C:/Users/Ant/AppData/Local/Temp/mnemos-merged-package-v2`. Its module origins
point at the installed target. The existing legacy environment's 35 package
files remained unchanged. Installed Python 3.11.13 runs passed 15 storage
regressions and three automated Tier 1 MCP flows (one non-Tier-1 test deselected).
The external wrapper/configuration copies match source. The consumer has no
mnemos source directory.

Operational probe v3 uses disposable synthetic SQLite data and an isolated home:
doctor reports expected degraded mock-provider readiness, counts one legacy
unscoped chunk, and reports scope isolation unready. Project-only MCP retrieval
and health pass. Startup and retrieval-latency events appear. A mocked HTTP 400
produces the provider-failure event. An injected graph insert failure raises
IntegrityError, leaves no open transaction and now emits one storage-failure
event with successful replacement rollback. Installed package bytes still
match the wheel after the probe. No real provider or interactive host was used.

Full-source evidence remains the prior 329 passing tests. On this unchanged
commit, complete Black and mypy checks pass for 66 and 31 files respectively.
Hosted CI run [42503225584](https://github.com/anthony-maio/mnemos/actions/runs/42503225584)
was confirmed in progress on this commit; its conclusion is not yet known. The previous private wheel
and failed operational v1 probe remain preserved, not overwritten or regraded.

Private archive `.venv/storage-merged-package-evidence-v2.zip` contains 28
evidence files plus a verified manifest, including build/Twine/lint logs,
artifacts, installed results/XML/logs, operational outputs, helpers and the draft
PR body. Every manifest entry was read back and verified. Archive SHA256:
`1398462d229bd5fefa7c1e1abf88abd001d3429c63f7299b7d59decbd29e2f52`.

Remaining release work includes hosted CI and the checklist's interactive
host and actual pilot requirements. No release requirement is waived by the
automated results. The original semantic quality, complete capture/recall
latency, native acceptance, pilot and broader delivery obligations remain open.
