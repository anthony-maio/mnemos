# Source association frozen vectors v1

Status: preparation complete and independently verified. Graph measurement
and quality acceptance are NOT RUN. These vectors do not supersede any
previous negative graph or retrieval result.

The [preparation protocol](../../../plans/2026-09-20-graph-association-vector-preparation-v1.md)
and five-case manifest were committed before the first model request at
`326322f02fc372adaced52dccf8b316c95cedc04`. The run lasted from
2026-09-20 09:39:05.735501 UTC to 09:39:15.621351 UTC. Source revision and both
input hashes were unchanged at completion.

## Observed preparation

- Endpoint: `http://127.0.0.1:11434`; Ollama reports version `0.34.1`.
- Model: `nomic-embed-text:latest`, F16, 137M, digest
  `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`.
- All query, document, and symmetric prefixes are empty. No prefixed quality
  comparison is implied.
- 364 unique texts, 364 role bindings, 23 embedding requests, 30 total API
  requests. Each vector has 768 finite dimensions and is nonzero.
- Batch size at most 16; truncation disabled; no retries or fallback.
- Final `/api/ps` reports context length 2048 and `size_vram=323150151`, equal
  to the reported model size. This is observed model residency, not a GPU
  performance benchmark.
- Model digest and effective `/api/show` metadata stayed stable.
- Every exported vector matches the corresponding raw response. Every raw
  request and response length/hash was independently verified. Exact input
  order, text coverage, role bindings, and input manifest hash also match.

## Bindings and checks

| Artifact | SHA-256 |
| --- | --- |
| `artifact.json` | `bb66da961d731c80a9bb965f72db082371f5faac2f13b32ff268bb408a7420cb` |
| Authored case manifest | `809a3a27ed0052e0fdf872542168af0a38337c1dc42f3d0b2af66d769cee5741` |
| Preparation helper | `a15716c9d50733db4900ca5051dcb545a5bad0ce4599b519b697e2d280027814` |

Before the run, 14 manifest/preparation tests passed, including real temporary
Git fixtures, preflight rejection before transport, metadata/digest drift,
invalid vectors, and failed request preservation. Black and strict Python
3.10 mypy passed for the helper and both test modules. The earlier diagnostic
and manifest suite separately passed 24 tests. These are scoped benchmark
checks; the latest complete product package remains source `5155ebd`.

`index.json` binds 60 evidence files, including raw API bodies, the vector
artifact, helpers, and the 14-test JUnit report. This README and the index are
outside that file list. The original temporary artifact is retained.

The next step is the separate three-arm measurement with production retrieval
and disposable relation views. No similarities, paths, target ranks, or arm
comparisons were calculated in this preparation. Retain all five cases
regardless of whether their targets become direct candidates.
