# Source-bound recovery preparation

Frozen product source: 11a97e34214a038e208c116d0778d0dc5681fcff.
The runner verifies all 71 source files before using the isolated snapshot.
It uses the original outage fixture and unchanged exactly-one-marker, scoped
recall, correction, restart, forgetting and replay assertions. All five module
flags remain enabled; the extraction mode is now source_bound.

Provider: local Ollama, Qwen3.5 9B, Nomic embeddings, temperature 0,
think=False, 8192 output-token limit, 180-second provider timeout. Exact model
digests and complete configuration are recorded in preparation.json.

The first harness launch failed before service startup or model calls because
an obsolete prerequisite check remained. That failure is preserved separately.
The corrected runner was launched against a fresh copy of the original fixture.
This directory records preparation only. It is not a passing lifecycle result,
installed-wheel validation, held-out quality evaluation or native-host capture.
No live host settings or memory data were changed.
