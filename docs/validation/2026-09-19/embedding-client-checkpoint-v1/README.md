# Embedding client reuse and shutdown checkpoint

Source `39b9c799020a58ea2ac3d798189474ed10389bd0` reuses one Ollama embedding HTTP client per provider.
TLS defaults, environment behavior, per-request timeouts, fallback, and retry
settings are retained. Concurrent requests share client initialization. Close
waits for active requests and other close calls, rejects new requests, permits
observer-triggered close, and preserves transport teardown failures.

Configured engines own their providers; library-injected providers remain
caller-owned by default. Shutdown drains one shared cleanup task even if a
caller cancels. Completed resources are not closed twice; failed cleanup stays
visible. The managed service stops and drains work before closing the engine.

Verification: 1123 source tests passed, with 4 skips.
Black, strict mypy, JavaScript syntax, build, and Twine checks passed. The
isolated wheel run passed its selected source-matched contract cases and
replay assertions; exact case outcomes, package origins, and before/after
hashes are retained under `installed-checks`. These are source/package checks,
not native-host or model-quality acceptance.

A separate disposable HTTP/1.0 fixture measured warm request median
200.344 ms on frozen `fc72cc2` and 1.540 ms on this checkpoint,
using 12 warm requests after one cold request per arm. The fixture closes every
TCP connection, returns a fixed vector, and makes no Ollama/model calls. This
isolates setup overhead; it does not close the 500 ms recall or 100 ms capture
gates. Those real-model gates remain failed pending a new bound measurement.
The async cognition provider's per-call HTTP client is unchanged.

The repository's separate synthetic replacement gate failed both repetitions
on both source snapshots. MRR lift ranges from -0.4 to 0.0 on both stores in
both snapshots. Thresholds were unchanged. Therefore this checkpoint is not
an all-CI-gates pass and supports no replacement/superiority claim. The first
benchmark invocation inherited existing OpenRouter configuration and failed
HTTP 401; its log is retained. The two completed comparisons explicitly used
simple embeddings and a missing configuration path in child environments.
No live configuration was changed.

The first source verification launch rejected a checkout still in progress
before running checks. The successful run used the completed clean snapshot
and fresh output/temp directories. Both setup failures are distinguished from
the actual benchmark failures in `checkpoint-binding.json`.

The original real-model matrix and its queued comparisons still use frozen
`fc72cc2`; this repair does not modify that source or those results. No release,
live host configuration, live memory database, or website was changed.
