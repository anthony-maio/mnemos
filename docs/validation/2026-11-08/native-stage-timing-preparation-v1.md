# Native stage timing preparation

Added an opt-in `stage-timing` build feature to the experimental native launcher. Default builds collect and emit no timing data. The diagnostic build emits one stderr record with fixed stage names and elapsed milliseconds, without payloads, paths, tokens, environment values or request identities. It preserves the normal stdout response and never adds a retry.

Marks cover preparation, stdin parsing, descriptor validation, serialization, connect, request write, first response byte, completed headers, body read, validation and output handling. Failures produce partial traces. A failed stderr write does not change the hook result. Process creation before main, timing-record output and process exit require an outer command timer.

Both release builds compiled with locked dependencies offline. All 80 Python launcher/profile/environment/diagnostic checks passed, including five new tests. Both Rust transport deadline tests passed with timing enabled. Formatting and diff checks passed. The checks cover valid dispatch for both hosts, preserved large-integer payloads, unchanged stdout, missing-worker fallback, lost-response no-retry behavior and sanitized invalid input.

The frozen [pipeline protocol](../../plans/2026-11-08-native-pipeline-diagnostic-v1.md) and isolated helper are prepared for the unchanged installed response-completion package. No measurement result or performance improvement is claimed by this preparation. The [receipt](native-stage-timing-preparation-v1.json) binds both binaries, tests, helper and protocol. Earlier binaries, packages, failed timing results and live pilot configuration are unchanged.
