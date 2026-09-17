# Isolated cross-host diagnostic

The single Codex parameter that failed in the full e1fd56e suite passed in
a fresh external fixture: one test, 4.32 seconds, all 40 traced requests
completed. The list response contained one fact and no degraded/error flag.
The original five-second client timeout and durability settings were retained.

This run did not reproduce the original missing-facts response. Its cause
remains unknown, and the full-suite failure remains open. This is a synthetic
adapter/service contract test, not a native Codex session. The diagnostic uses
an in-process request observer and importlib test loading; it does not recreate
the earlier full-suite execution environment or ordering.
