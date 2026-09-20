# Native Hermes v11: three failures at forgetting

The fixed cohort finished zero of three complete lifecycles on verified
source `697b6720b201d578c03ce33d98dff3dda6d2def8`, wheel SHA-256
`49c7d791cd6b5d7645125cec72371b453607543b371e6834c6a2e122774f42b0`.
Runner inputs were committed at `bf85eb8` before execution. The cohort ran
from 08:54:42 to 09:01:25 UTC on September 20, 2026. All three attempts
established cleanup and preserved the local control and foreign project.

Every attempt passed teaching, restarted recall, same-ID correction to
version 2, and restarted recall of the corrected fact. Each then failed at
forgetting. No attempt reached source replay or final restarted recall.

| Attempt | Forgetting result |
| --- | --- |
| 1 | Inspected the target and four related candidates, then selected and deleted two related assistant captures. The target was deleted and suppressed, but two unselected user captures still contained the release labels. The no-retained-label gate failed. |
| 2 | Inspected all ten related candidates, then selected all ten fact IDs with two source IDs that were absent from the preview. Two identical apply calls returned degraded HTTP 500 without a committed receipt. Acceptance stopped on the unsuccessful tool result. |
| 3 | Retrieved and inspected the target and obtained a preview, then ended without an apply call. The target and related copies remained. |

The attempt 2 request reused the preview's digest and limit correctly. Its
source selection differs from the preview's ten required source IDs. The
storage implementation rejects out-of-preview source IDs before mutation,
using an untyped ValueError that becomes HTTP 500. This is a source-grounded
explanation of the observed response; the native archive has no server
traceback. The model's claim that the error was transient is unsupported.

A separate [terminal database inspection](terminal-analysis.json) reads all
three databases without maintenance or writes. It confirms unchanged control
facts, events, and evidence; attempt 2 retains its target with no suppressions.
It also binds the source-selection comparison to the original trace bytes.

The [boundary analysis](boundary-analysis.json) verifies all 50 model
responses against recorded loaded-model observations: each used the pinned
Qwen alias, 65,536-token serving context, and positive GPU residency. Exact
provider request bodies, original tool responses, phase states, readiness
records, protocols, and input hashes are retained. The local model is
Qwen3.5:9b; embeddings are real Nomic. All five cognition modules are disabled.
This is a bounded native tool lifecycle, not an evaluation of full-module
memory quality or production latency.

The package's shared-guidance repair does not change these Hermes-only
instructions. This cohort does not install the shared Codex AGENTS file and
cannot measure that repair's effect. The earlier v10 result remains one
pass and two failures on its own package. Neither cohort is a causal
comparison, and their outcomes are not pooled into a reliability estimate.

The [v11 import preflight](../hermes-native-discovery-preflight-v11/README.md)
passed separately. The [package checkpoint](../shared-guidance-checkpoint-v1/README.md)
also remains passed. Native acceptance for this package is open; its planned
three-host v6 attempt has not run because this prerequisite has no full pass.
There are no selective reruns. Live host settings, source, and memory were
unchanged. This README is outside the raw evidence index.
