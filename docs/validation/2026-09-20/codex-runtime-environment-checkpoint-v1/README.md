# Codex runtime environment package checkpoint

Frozen product `9722d92a5003a5dd47b0bf7846cdeede1b5a77f7` passes 1,857 source tests
with 4 skips, all six source/package checks, 703
isolated installed cases with 4 matching skips, and seven replay
assertions. Installed checks include generated host configuration and compare
individual outcomes, imported module origins, and package bytes with the
frozen source and wheel.

Wheel: 305,120 bytes; SHA-256 `59160aa424f4ad090dd8c2232c5a32ac87b128f22b13af68a7b7d808af6b23f3`.

The [development record](../codex-runtime-environment-development-v1/README.md)
preserves two failing regression checks and 93 focused passes with two skips.
Codex MCP and hook configuration now accept explicit autostart, runtime, and
SQLite selectors. Disposable native phases supply those selectors so their
configuration does not depend solely on subprocess environment inheritance.
The allowlist continues to reject unrelated environment keys.

These checks establish package behavior. The subsequent [native Codex outage](../native-codex-outage-v1/README.md)
separately observes one raw-capture recovery on this package. The earlier [Claude outage](../native-claude-outage-v2/README.md)
is bound to product `a64afb7`; its raw capture result is unchanged. Broader
outage recovery, full-module quality, Desktop, and pilot gates remain open.

The evidence index binds copied raw records and validation helpers. This
README is outside that index. No live host configuration or database was
changed by this package validation.
