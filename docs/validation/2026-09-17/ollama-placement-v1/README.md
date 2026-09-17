# Ollama placement observation v1

This archive records one read-only placement snapshot taken on 2026-09-17 at
2026-09-17T22:27:43Z against the local Ollama endpoint
`http://192.168.86.35:11434`.

The observed Ollama version was 0.34.1. The `/api/ps` response reported
`nomic-embed-text:latest` (digest
`0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`, size
376449269) and `qwen3.5:9b` (digest
`6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`, size
6604298647), both with `size_vram: 0`. This indicates CPU placement as reported
by Ollama. Historical placement is unknown, and this snapshot
does not establish a power-loss, driver, or other cause.

The user reported two RTX 3090 cards with 24 GB each. That hardware report was
not remotely verified by this observation. Ollama's documentation describes
`/api/ps` as the list of models currently loaded in memory and documents GPU
support and placement controls:

- [List running models](https://docs.ollama.com/api/ps)
- [GPU support and placement](https://docs.ollama.com/gpu)

The baseline checkpoint had 600 completed observation IDs and 600 completed
observations. The full checkpoint had 3 completed observation IDs and 3
completed observations. These are counts only; no observation quality was
inspected. The generic graph and affective row names were absent from this
checkpoint schema.

The supplied coordinator PID 26488, lifecycle PID 47588, and child PID 46036
were present as Python processes when sampled. Their names and creation times
are retained in `report.json`.

The helper performs only bounded GET requests to `/api/ps` and
`/api/version`, process metadata collection, and checkpoint length reads. It
does not infer latency causes and does not perform inference, pull, unload,
configuration changes, or model lifecycle operations.
