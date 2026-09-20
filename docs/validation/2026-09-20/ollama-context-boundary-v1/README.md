# Ollama endpoint context mismatch and alias repair

The three scheduled tool-free probes confirmed an endpoint mismatch on
local Ollama 0.34.1:

| Model alias and endpoint | Requested context | Loaded context | Reported model VRAM bytes |
| --- | --- | --- | --- |
| greedy-v1, `/api/chat` | 65,536 | 65,536 | 8,044,454,869 |
| greedy-v1, `/v1/chat/completions` | 65,536 | 16,384 | 6,014,380,276 |
| greedy-ctx64k-v2, `/v1/chat/completions` | 65,536 | 65,536 | 8,044,454,869 |

The new alias pins `num_ctx: 65536` in its model parameters. Its weights,
template, model information, capabilities, and other sampling parameters
match the earlier alias; all pre-existing model digests remained unchanged.
No new weights were downloaded or live host settings edited.

This follows Ollama's [documented context configuration for the OpenAI endpoint](https://docs.ollama.com/api/openai-compatibility#setting-the-local-context-size),
checked September 20, 2026. A request field alone did not establish the
loaded context. `report.json` and the exact request/response files provide
the local evidence; the documentation explains the supported configuration.

## Consequences for earlier claims

The v1 readiness helper used `/api/chat` and checked positive GPU residency,
but did not require an exact loaded context. Hermes used the OpenAI endpoint.
The frozen request-profile gates checked the intended request field, not
the serving model's effective window. Keep that distinction in reports.

Earlier [single-turn host snapshots](../../2026-09-19/native-hermes-local-v4/ps-loaded.json)
also show 65,536 before the host, while [its post-host snapshot](../../2026-09-19/native-hermes-local-v4/ps-after.json)
shows 16,384 for the same Qwen base-model digest. Cohorts without per-phase
loaded-context observations cannot be relabeled as verified 65,536-token
workflows. Nor does this establish the exact window in every historical
phase. Their failures and bounded observations remain unchanged.

This is a verified configuration defect, not proof that context size caused
all missing tool calls or malformed forget arguments. The short failed teach
requests particularly require their own explanation. The replacement
[model profile](../native-model-profile-v2/profile.json) has only endpoint
context-probe evidence at creation. The later [native v9 cohort](../hermes-native-cohort-v9/README.md) verified context after every response but failed all three lifecycles; native memory acceptance remains open.

## Provenance

The [protocol](../../../plans/2026-09-20-ollama-context-boundary-protocol-v1.md)
and corrected runner were committed before the three requests. The initial
launch failed in request logging before any HTTP call; its zero-request
report and original runner are retained alongside the corrected version.
No inference case was retried. The corrected run binds commit
`1e716770ce571348bc38bf0cd35e65f7b721657c` and finished with contexts
`[65536, 16384, 65536]`. The evidence index binds 26 raw files; this README is
outside that index. This diagnostic does not close native lifecycle,
three-host, model-quality, or pilot acceptance.
