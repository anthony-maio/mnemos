# Native Codex shutdown diagnosis

The interactive v2 process reached its 180-second watchdog after a successful
MCP retrieval. Selected native host log rows show shutdown beginning at
23:15:55 UTC and the Mnemos MCP child exiting with code 1 at 23:15:56 UTC.
Codex recorded the Mnemos transport as closed. The parent process nevertheless
remained alive until the watchdog. This narrows the observed hang to work
remaining after that child exited; the parent hang and child exit-code causes
are not established.

A separate installed-package control initialized the MCP stdio server, sent
the initialized notification, then closed stdin. The server exited with code
0 in 74.425 ms and emitted no stderr. All 56 installed package files matched
the source30eaf42 wheel. This control did not call a service tool, launch a
native host, or run model inference. It does not reproduce the native shutdown
sequence or establish that all MCP shutdown paths are reliable.

The selected log observations retain row IDs, timestamps, targets, and hashes
of the original bodies. They are derived sanitized observations, not complete
raw logs. Authentication, HTTP headers, prompts, and the private log database
are excluded. The original interactive report remains failed.
