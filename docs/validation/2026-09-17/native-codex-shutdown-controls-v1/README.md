# Native Codex shutdown controls

Codex 0.146.0 exited zero in two disposable controls with no configured Mnemos hooks, MCP server, or service. The first exited after idle startup; the second completed one native model response without tools before exit. Both used read-only unelevated sandboxes. The first required interactive sandbox setup; the second configured that same mode before launch. Neither process exceeded the unchanged 180-second limit, which includes setup and terminal interaction. These are one observation per condition, not latency measurements.

In contrast, both native continuity v4 sessions exited 0xc0000005 after successful MCP operations. Two Windows Application Error records in their shutdown time window identify codex.exe and fault offset 0x000000000cba0dde. The continuity runner did not retain child PIDs, so the event timing is corroborating evidence rather than an exact PID binding.

The controls rule out an unconditional idle or simple-response shutdown crash in this executable. They do not isolate MCP transport, hooks, tool approval, or their interaction as the cause. The integration failure remains open; no host upgrade or live configuration change was made. Authentication copies were absent before archiving.
