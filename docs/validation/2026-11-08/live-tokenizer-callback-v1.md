# Live tokenizer callback boundaries

The benchmark callback now sends each complete candidate context directly to
the separately configured native tokenizer. It has no count cache, accepts only
a literal loopback HTTP origin with an explicit port, disables redirects and
environment proxies, and sends `add_special=false` and `parse_special=false`.
It retains each request, raw response, count, exception type and elapsed time
in memory for the measurement runner to persist outside the timed callback.

HTTPX uses a two-second timeout for each network operation. This is not a hard
two-second wall-clock deadline for the entire request. The runner must retain
complete elapsed time, including a failed callback and subsequent repacking.

Fifteen new checks and thirteen existing context-budget checks pass together:
28 passed in 0.71 seconds. They cover uncached repeated requests, literal Unicode
and special-token syntax, malformed token IDs, timeout propagation, whole-context
byte fallback after an earlier record was skipped, endpoint restrictions and
client closure. The fallback test verifies selected records and context against
ordinary byte packing, not merely an error label.

These checks use a mock HTTP transport. They do not establish a live tokenizer
response, actual network deadline behavior, Qwen compatibility or retrieval
latency. Black formatted both changed files, scoped mypy passes for the callback
module with `--follow-imports=silent`, and `git diff --check` passes.

Only benchmark code and tests changed. Production packing, canonical product
source, live configuration, databases and frozen results remain unchanged. The
[integration protocol](../../plans/2026-11-08-live-tokenizer-retrieval-v1.md)
still requires the runner, artifact freeze, 408 scheduled observations and
independent audit before any integration result can be reported.
