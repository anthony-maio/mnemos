# Supplemental comparison v2 launch

The independent encoding/retrieval queue started on September 19 at
19:50:51 UTC. The preserved 20:06:49 UTC observation confirmed the coordinator
and encoding worker were live and the full encoding arm had completed 97
observations. This is a dated launch record, not a final result or a claim
that the process is still running when this document is read.

Bindings remain `fc72cc251aeafa161a1a390269b8dfa34d83cf2b`, the held-out v3
dataset and original comparison protocols, Qwen3.5:9b, and Nomic embeddings.
Exact source/dataset/protocol/model hashes and commands are in the archived
coordinator and state. The endpoint is the authorized replacement local Ollama
server, `http://127.0.0.1:11434`, on the RTX 4080 machine.

Encoding runs three arms with three repetitions over 200 held-out sessions.
Retrieval follows with four strategies and three repetitions. An exit-zero
`incomplete` manifest is preserved and does not prevent the independent next
comparison. Nonzero exits, failed/unknown manifests, or missing artifacts stop
the queue for inspection. This changes coordination, not labels, observations,
quality thresholds, or acceptance gates. The previous queue's failure and its
unstarted comparisons remain preserved.

CPU-only regression and package work ran concurrently. No other model job was
intentionally launched. Preserve that load qualification with latency results;
these comparisons are not isolated latency benchmarks.

The runtime state is in
`C:/Users/Ant/AppData/Local/Temp/mnemos-local-supplemental-v2/state.json`.
Before waiting or restarting, inspect the actual recorded process/session.
`evidence-index.json` binds the five launch artifacts. This README is outside
that raw index. The included preparation note predates execution.
