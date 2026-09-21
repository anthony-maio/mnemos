# Source distribution validation payload exclusions

Source 232d0da excludes validation ZIP files, SQLite snapshots/sidecars and raw
provider dumps from the source distribution. Historical repository evidence is
unchanged. CONTRIBUTING explains that complete evidence manifests must be checked
against the repository at the recorded commit, not the trimmed distribution.

The new source archive is 40,653,239 bytes, compared with
399,753,409 bytes in the preceding a128ef7 build.
This removes 359,100,170 compressed bytes, about 90 percent.
No excluded payloads occur in the new tar member list. The exact contract dataset
and all 14 native trace phases plus their attempt reports retain their bytes.
All 50 tests in the two archive-dependent test files pass from the extracted
source distribution, with no skips.

Building a wheel from that extracted source succeeds and produces the exact
same 310,407-byte wheel as the preceding fully verified build, SHA-256
ab04113f7fe6a69d45c88d99e33ee8bd31245a596720a7ecbc38ab23cf336a30.
Package metadata checks pass for both artifacts. The 2,158-test full source suite
remains bound to a128ef7; this is a packaging-only change with extracted-source
fixture tests and byte-verified wheel equivalence, not a new full-suite claim.
The previous installed and Hermes v36 checks cover the identical wheel bytes.

No release was published, live configuration changed, memory migrated, or
historical negative result removed. Semantic extraction and T17-T19 acceptance
remain open. Model experiments keep their original frozen source bindings.
