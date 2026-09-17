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

`prefetch()` injects bounded baseline recall. `sync_turn()` queues user and
assistant evidence without blocking the Hermes turn. The local service or its
durable outbox acknowledges each accepted event. Repeated callbacks use stable
event identities and do not create another observation.

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
attempts have not completed a conversation.
