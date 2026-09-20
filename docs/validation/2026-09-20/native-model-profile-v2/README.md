# Native Hermes model profile v2

`mnemos-native-qwen35-9b:greedy-ctx64k-v2` reuses the prior greedy-v1 weights
and sampling settings, adding only the model parameter `num_ctx: 65536`.
Its digest is `87bc5156d86918e0435431ab0286167a4c79c4a4a259b82fba5f3f75022006e2`.

The [endpoint diagnostic](../ollama-context-boundary-v1/README.md) verified
65,536 loaded tokens and positive reported GPU residency after an actual
OpenAI-compatible request. The old alias remains unchanged. This repairs
the configuration boundary observed when the old request-level context
option failed to govern that endpoint.

`profile.json` records the parameters, lineage, unchanged weights, and probe
binding. It does not claim a successful native lifecycle or improved tool
selection. A separately frozen successor must verify this profile through
the actual serving endpoint and inspect loaded context during the workload.
