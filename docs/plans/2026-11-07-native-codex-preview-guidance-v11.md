# Native Codex preview guidance validation v11

Run one fresh six-phase attempt on product `c80eb051`. It changes the shipped
host instructions and MCP preview metadata. It does not change dispatch,
storage, selection, version checks or the native acceptance rules.

V10's original requests exposed a guidance gap: the host read the advertised
names but did not inspect the preview definition before either missing-version
request. Its generated instructions explained version handling for correction
without stating how to begin a forget preview. Preserve v10's failed result,
its successful inspection calls and its eventual deletion evidence unchanged.

The new instructions require reading each tool's current definition before
first use. The Codex artifact explains that Code Mode discovery includes the
matching ALL_TOOLS description. The shared workflow says to inspect the target,
copy fact.version into the preview's required expected_version, review all pages,
and use unchanged apply_preconditions with explicit selections. It also explains
stale previews and checking error responses before reading successful results.
The MCP parameter remains a required strict integer and now identifies its
source in the advertised schema and opening description.

Before inference, require 95 focused passes across transport, stale-state
protection, scope, host integration, CLI generation and inspection. Two POSIX
hook execution checks are skipped on Windows. Require strict mypy on both
changed modules, Black on all three changed files, and the captured generated
artifacts and actual FastMCP definition. The unchanged inventory/lifecycle
harness retains 179 focused passes. The earlier 3,495-pass full suite remains
bound to its own source.

Build and record a new wheel hash, check metadata, run the real-embedding
installed CLI/MCP restart-recall smoke, and compare installed bytes with the
wheel and committed source. Use a new external native environment with the same
57 dependency versions and unchanged 17 embedding assets. Confirm its generated
Codex instructions match the reviewed artifact. Use fresh homes, project roots,
registry, database and output. No existing host configuration is applied.

Keep the v9 complete inventory observation and original six lifecycle prompt
hashes. Do not add version hints to the test requests. Keep the pinned Codex
CLI 0.164.0-alpha.11.1 and Code Mode binaries, gpt-6-astra, high reasoning effort
and 180-second phase limits. One bounded initial inventory readiness observation
is allowed as before. Any failed call, failed cell or other phase failure ends
the attempt; no phase retry or corrective message is permitted.

Retain original job completion, evidence inspection, same-ID correction,
reviewed deletion, exact source replay suppression, retained/foreign controls
and final native recall checks. Audit the native IDs and complete inventories.
Also verify each native phase received the installed generated instructions;
keep prompt-file byte hashes separate from exact native prompt-string hashes
to account for Windows CRLF. Auditing must not repeat inference or alter scores.

Require at least one hour on the login, preserve normal credentials and
configuration, remove copies and runtime descriptors, and verify owned process
identities have exited. Keep raw streams private. This remains a synthetic
workflow attempt, not a daily-use pilot, quality result, hook latency acceptance,
live configuration change or publication.
