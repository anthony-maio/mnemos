# Local model readiness, September 19, 2026

The old `gpt-oss:20b` tag could be listed but `/api/show` failed with HTTP 500
and a GGUF tensor-size overflow. Before refreshing it, the old manifest was
copied to `mnemos-gptoss20b-before-refresh:20260919-v1`; the copied digest
matches the old `aa4295ac...` digest in the preserved API response.

The refresh downloaded a different model blob. Its client stopped with a
read timeout while Ollama was verifying SHA-256. This failed client record
is unchanged. A later read-only tags/show check found the new manifest
`17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7`
and HTTP 200 metadata. The server completed after the client stopped; no
second pull was launched.

A single separate load request used low reasoning, temperature 0, seed 42,
32,768 context, and a 128-token output cap. It returned HTTP 200, `done: true`,
and `READY.` in 37.06 seconds, including reported load time of 19.16 seconds.
The helper's exact `READY` assertion rejected the period, so its terminal
status remains failed. This is a helper assertion failure, not a provider
load failure. No replacement request was run. A subsequent read-only
`/api/ps` response reports 12,765,573,938 model bytes, all in VRAM, and a
32,768-token context. `nvidia-smi` identified the RTX 4080 with 16,376 MiB
total and 15,526 MiB used at that observation. This is a load observation,
not a sustained utilization or tool-quality result.

The raw folder retains successful and failed calls, the streaming pull
frames, the fixed readiness request and response, and follow-up reads.
The refresh, load, and archive helpers are preserved. The three small
read-only API observation commands were run interactively; their raw
responses are present, but they have no standalone helper artifact.
Times in JSON are UTC (September 20); the folder date follows local time.

The [development protocol](../../../plans/2026-09-19-local-model-forget-development-v1.md)
and [profiles](../../../plans/2026-09-19-local-model-forget-profiles-v1.json)
are separate prospective inputs. No native-host, forgetting, or module
acceptance follows from this readiness check. No live host configuration
or memory database was changed.
