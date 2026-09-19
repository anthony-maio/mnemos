# Supplemental comparison coordinator v2

This coordinator is prepared but has not been executed. It is independent of
the failed `fc72cc2` lifecycle queue and never reuses its output, work, or
runtime directories.

Bindings remain frozen to source `fc72cc251aeafa161a1a390269b8dfa34d83cf2b`,
the held-out v3 dataset SHA256
`036fa51ee13271a85a14a62a2fe4ef65d13eaeb290331fe325aa5bf32ce893fc`,
Qwen `qwen3.5:9b` SHA256
`6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`, and
Nomic `nomic-embed-text:latest` SHA256
`0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`.
The protocol hashes are recorded in the coordinator itself and are checked
before each child stage.

The encoding stage runs first with the three protocol arms and three
repetitions over 200 held-out sessions: 1,800 arm/session observations. The
retrieval stage then runs independently with four strategies, three
repetitions, 300 query identities per repetition, and 3,600 strategy result
rows. Both stages use top-k 5, a 1,200-token context limit, temperature 0,
512 maximum output tokens, 120-second provider timeouts, and explicit
`--think false`. The child binding also records the unchanged Qwen model,
Nomic embedding model, three repetitions, and these options in the v2 runtime
state before each child is launched.

Only a child execution failure, failed manifest, missing artifact, or provider
transport failure stops the sequence. The frozen runners use `completed` for a
fully covered stage and `incomplete` when the child ran but its coverage or
quality outcome is incomplete. An exit-zero `incomplete` manifest is recorded
as terminal `incomplete` and does not prevent the other comparison from
running; a negative quality gate alone is never treated as a transport error.
The coordinator records each stage command, logs, manifest status, exit code,
and terminal result immediately. Expected cost is unknown until the child
reports provider usage; no cost is estimated here.

Fresh v2 paths:

- Runtime: `C:\Users\Ant\AppData\Local\Temp\mnemos-local-supplemental-v2`
- Encoding output/work: `docs/validation/2026-09-19/encoding-comparison-local-v2` and `C:\Users\Ant\AppData\Local\Temp\mnemos-encoding-comparison-local-v2`
- Retrieval output/work: `docs/validation/2026-09-19/retrieval-comparison-local-v2` and `C:\Users\Ant\AppData\Local\Temp\mnemos-retrieval-comparison-local-v2`

No model, GPU, or network request was made while preparing these files.
