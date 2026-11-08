# Distinct legacy prerelease artifact

The combined legacy candidate now has a distinct unpublished identity,
0.6.1rc1, at c36ca712c47e9c8c0f46226e74ad805fd7be0cef. Package declaration,
runtime version, Claude plugin and marketplace manifests agree. The version
check found 0.6.0 as the latest published package in [PyPI metadata](https://pypi.org/pypi/mnemos-memory/json).
The candidate is not uploaded, tagged or activated.

The README and release documents now state this candidate's limits instead of
inheriting broad release-ready and interactive-host claims. The upgrade guide
explains schema 3, stopping older clients, consistent backup, copy-based rehearsal
and the limits of rollback after new writes or forget requests. ClawHub publishing
instructions no longer suggest publishing an unreviewed candidate as latest.
All 25 metadata and documentation checks pass on Python 3.12.11.

Isolated Hatchling 1.33.0 builds the source archive, then builds the wheel from
it. Both pass Twine. All 35 package files match source, source archive, wheel
and external installed target; RECORD digests and lengths pass. The packaged
upgrade guide and manifests match source. Runtime and installed metadata both
report 0.6.1rc1.

In a separate consumer directory, 39 unchanged storage and recency tests pass
on Python 3.11.13. They include transaction rollback, another process deleting
a source, caller commit/rollback with FTS edits, content/access separation,
legacy migration and concurrent opening of an existing WAL database. Three
additional MCP roundtrips pass using the generic, Claude Desktop sample and
Claude plugin configurations with the exact copied wrapper. The consumer has
no source mnemos directory; package origins point to the installed target.
Mock LLM and simple embeddings keep those MCP tests local.

An independent XML audit verifies all 67 checks across their two environments.
The full 353-test source result remains bound to the preceding implementation
commit b3508f6d, whose product calculations are unchanged by the metadata and
documentation update. The new installed checks qualify this exact artifact;
they do not transfer a supported-platform matrix or native-host acceptance.

The [receipt](legacy-integrated-rc1-v1-receipt.json) records exact hashes and the
private archive, which retains the artifacts, patch, review draft, helpers,
raw logs, JUnit records, package origins and dependency versions. The review
packet includes execution and structure diagrams. Earlier candidates, failures
and archives remain intact. The candidate worktree is clean and all launched
checks are terminal.

The original legacy benchmark failure remains unresolved; controlled chronology
results are development evidence on known questions. Quality, complete-command
latency, platform coverage, interactive hosts, actual pilot and publication
requirements remain open. No live database or normal host configuration changed.
