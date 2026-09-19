# Native Hermes local v4: bounded pass

All 16 checks passed for a single native Hermes turn: explicit session scope,
retrieval of the seeded fact, exclusion of a foreign-project canary, the
expected answer, and durable raw conversation capture. Four bridge capture
calls returned `committed`; exactly two raw events persisted, one user and
one assistant. The native turn took 24.592 seconds. A separate cold/load
preflight took 3.569 seconds; these are single observations, not latency tails.

Mnemos source: `78c63ed36942848c95c1acd93b805fe0e0d13a67`.
Wheel SHA-256: `4a8d364d69e69c64d43f876ffeb7b9b9b413efc997ad2b4cb8759ca0e15a9625`.
See the [package checkpoint](../hermes-continuity-checkpoint-v1/README.md).

Hermes 0.21.3 used an isolated checkout of
`14a346345486418aa85ed6526549412ee2665ade` with the proposed
[CLI session-directory patch](../../../integrations/hermes-cli-session-cwd.patch).
The installed host and live configuration were unchanged. Loaded host module
paths were checked against that isolated checkout. This does not establish
compatibility with the unmodified installed CLI, which omits provider cwd.

Local Ollama 0.34.1 ran Qwen3.5:9b and Nomic embeddings on the RTX 4080.
Exact model digests, effective context/reasoning settings, host patch hash,
callbacks, persisted events, and GPU observations are retained in this archive.
All cognition modules were disabled and seed maintenance completed before the
turn. This tests baseline recall and raw capture, not extraction quality.

The [v3 failure](../native-hermes-local-v3/README.md) remains preserved:
duplicate callbacks changed timestamps and were correctly rejected by the
outbox. V4 uses the repaired adapter without weakening collision checks.
Native correction/forget, outage/restart recovery, complete capture across
failures, and a two-week pilot remain open. Built-in remove/replace notifications
still lack exact Mnemos targets.

`evidence-index.json` binds 20 raw artifacts by byte count and SHA-256.
This README is outside that raw index. No live database or runtime credentials
are included.
