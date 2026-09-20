# Copyable forget preconditions and MCP description

Product checkpoint: `c7b2e65`. Status: six focused tests and 108 adjacent tests
pass. Full package and installed verification are separate checks.

The [design](../../../plans/2026-09-20-forget-apply-preconditions.md) responds
to the native v13 limit mismatch by returning `apply_preconditions` under
the four exact request names. It generates no selection and changes no
transactional checks. Incomplete previews return null preconditions. Hermes
and MCP descriptions explain how to use them; existing clients can continue
with the original fields. There is no new persistent state or migration.

Source inspection also found that MCP advertised preview as single-fact
deletion. A stray first docstring hid the intended preview description. The
registered FastMCP tool now describes a read-only preview correctly.

## Recorded checks

The first red report has five missing-field failures before implementation.
The second adds the reproduced registered-description failure. The green
reports each have six passes; the later report additionally checks source
removal and persisted suppressions on successful apply.

The tests copy the returned fields through the real Hermes adapter at limits
100 and 37, then add reviewed selections and verify deletion. A changed limit
or new matching capture after preview still rejects atomically, retaining
the facts and creating no suppressions. The adjacent suite covers storage,
selection/dependency safety, scope, service rejections, outbox behavior,
transport, MCP annotations, and documentation consistency: 108 passes.

Black passes for the changed Python files, and strict Python 3.10 mypy passes
for the three changed production modules. Full-check logs are recorded with
the package checkpoint rather than inferred from these targeted checks.

## Limits

Copyable fields remove a naming/placement translation step; they do not force
a model to copy values or choose valid IDs. The third native v13 failure,
which selected IDs outside the preview, remains outside this intervention.
An opaque review handle is a possible later design, with distinct state or
parsing and compatibility requirements. These deterministic tests do not
establish improved native reliability. No benchmark repairs arguments or
relaxes scope, digest, version, selection, dependency, or erasure requirements.

The index binds raw JUnit reports. Their source behavior is unchanged by the
follow-up commit normalizing three edited files to LF. The original native
failures and all quality gates remain open until separately measured.
