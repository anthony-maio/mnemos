# Current-candidate Claude lifecycle validation

Prepare one six-phase attempt against canonical candidate
`ec4a36b262be61c16d807a2b39006bcad7f8367c`, using the existing
`scripts/check-native-explicit.py` acceptance workflow. Preserve its teach,
restart recall, correction, corrected recall, reviewed forgetting and final
restart/replay checks. Preserve the scenario, phase prompts, source selection
requirements, foreign/unrelated controls, and original-source replay checks.
The earlier Claude v5 result remains bound to its older artifact.

The candidate wheel SHA-256 is
`ff7e97e1ed24a7122dd4094b9dcdd4ed2c5ece3e805ea3becca58e0e62502d5d`.
Before inference, prepare and verify an external installation and installed
smoke receipt matching this wheel. The registry-check installation inside the
checkout does not satisfy the lifecycle runner's external-installation rule.
Retain the 57 pinned dependency versions and do not modify the timing runtime.

Use the reviewed Claude 2.1.191 executable and six fresh print-mode sessions,
with 180 seconds and $0.50 per phase, total at most $3.00. No failed-phase retry
or corrective follow-up is part of this attempt. Keep one initialization report
per fresh single-input session, exact advertised phase catalogs, strict tool
correlation and queued-versus-completed checks.

Apply `claude_isolation.py` before the version probe and every phase launch.
Verify the reviewed binary hash, remove alternate inherited authentication
selectors, redirect profile paths, and disable CLAUDE.md discovery and automatic
memory without `--bare`, which disables OAuth. Preserve the existing explicit
plugin/hook workflow. Protect disposable directories and verify copied
credential removal and owned-process cleanup on every outcome.

Preparation status: isolation integration and 67 focused tests pass. The first
test run had six fixture failures because the new binary guard correctly
rejected nonexistent mocked executables; the fixtures now mock that guard only
while testing real environment isolation. No lifecycle inference has run.
Static review, external package preparation, frozen runner commit, full
regression and bounded outer supervision remain required before execution.

The preceding revocation evidence and its original helper sources were archived
before changing shared lifecycle helpers. Private archive:
`.venv/registry-claude-session-v1-evidence.zip`, 39 evidence members plus manifest,
108230 bytes, SHA-256
`8284d16e443a249cdc0819ae6994c410887efb94f5db39aa23d825e8c3162526`.
Use those bound sources when reproducing that historical audit.

Passing this attempt would establish the tested native lifecycle on the current
candidate, not quality, latency, pilot days, Desktop acceptance or publication.
