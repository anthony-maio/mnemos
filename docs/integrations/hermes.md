# Mnemos with Hermes Agent

Mnemos can run as an optional Hermes memory provider. The provider keeps the
Hermes built-in memory tools available and adds scoped Mnemos recall and
durable conversation capture through the local Mnemos service.

Hermes selects one external memory provider with `memory.provider: mnemos` in
the active `$HERMES_HOME/config.yaml`. The Mnemos package must be installed in
the Hermes Python environment through the `hermes_agent.memory_providers`
entry point. The provider is optional. If Hermes or its provider API is not
installed, Mnemos reports the manual MCP path instead of pretending that the
native integration is active.

Configure a registered identity before enabling the provider:

```text
MNEMOS_USER_PROFILE_ID=alice
MNEMOS_SCOPE_REGISTRY=/path/to/scopes.json
```

Provider registration reads this registry path before initialization. The
file must exist and contain a valid scope registry; a missing or malformed
file leaves the provider unavailable. An explicitly supplied constructor path
takes precedence over the environment value. Registration performs no network
request and does not by itself prove the memory service is reachable.

The provider receives the active Hermes `cwd`, `hermes_home`, and
`agent_identity` during initialization. It resolves project membership from
that session context. It does not use the process working directory as a
conversation identity and it does not silently join a different profile.

Initialization clears previously resolved scope. Every ordinary prefetch reads
the current service result so another client's correction or deletion is visible.
If the host
omits `cwd`, later callbacks remain unable to retrieve or capture. The installed
September 19 Hermes CLI does omit it, including when `--in` is supplied. The
[native attempts](../validation/2026-09-19/native-hermes-local-v1/README.md)
record this host propagation gap; native acceptance remains open.

The proposed [CLI compatibility patch](hermes-cli-session-cwd.patch) passes the
directory selected by the CLI into its agent constructor. It applies to Hermes
`14a346345486418aa85ed6526549412ee2665ade` and has not been applied to the live
installation. It does not repair gateway or other construction paths. Keep
patched-host evidence separate from support in an unmodified installation.

A separate proposed [durable message identity patch](hermes-durable-message-identity.patch)
preserves row IDs through late CLI resume, compression-recovery loads, and
undo/retry of an ID-less warm transcript. It targets the same pinned host
revision and has not been applied to the live installation or the cwd-only
native-test checkout. Its [28-case verification](../validation/2026-09-19/hermes-host-identity-candidate-v1/README.md)
includes real resume/rewind persistence and bounded loader contract checks.
Compaction that clones original rows still needs an occurrence-provenance
contract; carrying IDs through a reload does not solve that separate problem.

`prefetch()` injects bounded baseline recall. `sync_turn()` queues user and
assistant evidence without blocking the Hermes turn. The local service or its
durable outbox acknowledges each accepted event. Repeated callbacks use stable
event identities and do not create another observation.

When Hermes supplies a complete set of durable message `_row_id` values,
Mnemos binds each occurrence to its scoped native row rather than its text or
position. A regenerated turn gets new IDs; retained rows keep their IDs across
shortened views, checkpoints, and provider restart. Reusing a native row ID
with changed role or content is a payload conflict, not a new observation.
Malformed or duplicate row IDs reject the snapshot. Partially identified
snapshots are retained as provisional evidence, ineligible for fact extraction,
with a `partial_native_identity` diagnostic.

Callbacks with no native row IDs retain the older cumulative-prefix identity.
Appending later messages leaves those prior identities unchanged; repeated
identical messages at different positions remain distinct. That fallback has
the rewind limitation below. A private `mnemos/occurrences.sqlite` ledger
under the supplied Hermes home retains first-observation timestamps across
provider restarts. It stores event hashes and timestamps, without message text.
Preserve this ledger with the Hermes home and pending outbox during recovery.
Corrupt or unavailable state fails capture rather than inventing a new time.

Callbacks without a transcript have ambiguous occurrence identity. Their
content is retained as provisional evidence, ineligible for fact extraction,
with a `transcript_identity_unavailable` diagnostic. Each fallback callback
gets a distinct receipt; only replay of its retained payload is idempotent.
Several receipts may describe the same actual host turn, so receipt counts
do not prove transcript coverage or independent evidence.
An existing provisional receipt cannot be promoted into extractable evidence.
A later complete transcript can supply a separate normal occurrence, including
the same text. Identity hashes bind the supplied fields; they are not signatures
proving host authorship. The authenticated adapter remains trusted to submit
accurate host evidence.
Hosts omitting Hermes home also report `nonpersistent_timestamp`; timestamp
recovery is then limited to the provider lifetime. A
[same-session rewind diagnostic](../validation/2026-09-19/hermes-rewind-diagnostic-v2/README.md)
reproduces an identity collision in the older prefix-only path: after `/undo`,
an identical regenerated turn reuses the removed turn's IDs and timestamps.
Four queued observations remain
four where six logical observations are expected. A new session ID avoids
that collision. This is installed MemoryManager API evidence; the interactive
command and service persistence were not exercised. Native-row support in
`3730fd9` addresses this boundary when the host supplies those IDs; hosts that
omit them retain the limitation. Some late-resume and compression-adoption
loaders in the inspected host omit row IDs and need separate compatibility
work. Compaction can also clone rows, so native row identity alone does not
prove that retained copies count as one original observation across every
compression path. No live history or identity-format migration has been
performed.

Built-in memory `add` notifications can be mirrored. `remove` and `replace`
notifications do not contain verified Mnemos fact IDs and versions, so the
provider does not turn them into new captures or guess a mutation target.
`get_lifecycle_diagnostics()` exposes up to 32 content-free notifications for
unresolved actions and capture failures. These diagnostics are in-memory only.
`_flush()` reports pending capture failures; a repeated failed add can retry.
Automatic correction and forgetting across the two memory stores still need
an exact identity mapping and native validation.

The provider exposes `mnemos_remember`, `mnemos_retrieve`, `mnemos_inspect`,
`mnemos_correct`, `mnemos_forget`, and `mnemos_job_status` through Hermes'
native memory tool dispatcher. The initialized session supplies scope; tool
arguments cannot override project, profile, host, or session. Retrieval defaults
to the project. Correction and forgetting require an exact fact ID and its
current version. Inspect the fact before changing it.

Remember and correct also require a `request_id`. Use a fresh value for each
logical write and reuse it when retrying that same write after an uncertain
response. Remember and correction retries resolve to the original event/job.
A correction retry keeps its original expected version, even if processing
or a later correction has advanced the fact. Changing the content, target,
or expected version under the same request identity is rejected. A new edit
still requires the current version, and forgotten or inaccessible targets
cannot be retried. A committed response acknowledges durable evidence; job
completion establishes processing. When a committed write returns a `job_id`,
call `mnemos_job_status` once with that exact ID before the final answer and
report the observed status. Pending or leased work is not completed. The
provider's system guidance and write-tool descriptions both state this
follow-up; a native trace must still demonstrate that the host performs it.
Inspect the fact and job status after an
uncertain response, especially when multiple corrections were queued against
the same version.
These explicit tools do not perform fuzzy matching or bulk deletion.

The provider advertises the Hermes version 2 pre-compress checkpoint contract.
With checkpoint enforcement enabled by an operator, a failed or unavailable
Mnemos service raises before lossy compression. The default best-effort
setting remains unchanged. The provider does not select or replace Hermes'
compressor.

On Windows with Hermes in WSL, use an explicit Windows runtime descriptor when
it is visible to the WSL process, or configure the one-request bridge with
`MNEMOS_BRIDGE_COMMAND`. The value must be a JSON argv array, for example:

```text
["/mnt/c/Users/Ant/AppData/Local/Programs/Python/Python312/python.exe", "-m", "mnemos.bridge", "--descriptor", "C:\\Users\\Ant\\AppData\\Roaming\\Mnemos\\runtime.json"]
```

The bridge reads one bounded JSON request and writes one JSON response. It uses
the authenticated local client on Windows and never opens a listener of its
own. Do not expose the service on a LAN address, invoke a shell, or open the
SQLite database directly across the Windows and WSL boundary. The adapter does
not guess a WSL host address or modify firewall rules. Registry paths and
project aliases must be explicitly configured for the environment that reads
them.

Use `preview_host_integration("hermes", ...)` before changing Hermes configuration:

```text
from mnemos.hosts import preview_host_integration
preview_host_integration("hermes", mnemos_config_path="...", home="...")
```

The preview changes only the `memory.provider` selector. Provider paths and
profile values remain operator configuration, and the original config is
backed up when the change is applied.

Full Hermes acceptance remains pending until a real Hermes session demonstrates
scoped capture, recall, restart recovery, service outage buffering, and
cross-host continuity. Installed API, Windows/WSL bridge, and disposable plugin
discovery checks are recorded separately in the
[compatibility matrix](../client-compatibility-matrix.md); the native model
v4 run passes bounded session authority, recall, and raw capture with an
isolated CLI patch. The complete native lifecycle contract remains open.

The [installed API recovery probe](../validation/2026-09-19/hermes-api-recovery-v1/README.md)
separately verifies buffering during an outage, ordinary-request outbox drain
after explicit restoration, timestamp identity across fresh host processes,
and forget suppression across restart and replay. Its required checkpoint
success means evidence is durable in the outbox. The first retrieval that
drains that queue can still return no facts; maintenance and a later retrieval
are needed to observe extracted memory. This fixture-provider check does not
establish full native model lifecycle or automatic daemon restoration.
