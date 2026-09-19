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

`prefetch()` injects bounded baseline recall. `sync_turn()` queues user and
assistant evidence without blocking the Hermes turn. The local service or its
durable outbox acknowledges each accepted event. Repeated callbacks use stable
event identities and do not create another observation.

Repeated callbacks within one provider instance reuse the event's original
capture timestamp, keeping concurrent outbox payloads identical. The timestamp
map is provider-lifetime state; recovery of that timestamp in a newly created
provider is not established by the native single-turn fixture. Durable replay
continues to use the payload retained in the outbox.

Built-in memory `add` notifications can be mirrored. `remove` and `replace`
notifications do not contain verified Mnemos fact IDs and versions, so the
provider does not turn them into new captures or guess a mutation target.
`get_lifecycle_diagnostics()` exposes up to 32 content-free notifications for
unresolved actions and capture failures. These diagnostics are in-memory only.
`_flush()` reports pending capture failures; a repeated failed add can retry.
Automatic correction and forgetting across the two memory stores still need
an exact identity mapping and native validation.

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
attempts now complete conversations, but have not passed session authority and
the complete lifecycle contract.
