# Native Hermes retrieval-view cohort v8

The fixed three-attempt cohort completed with zero full lifecycle passes on
the verified retrieval-view package. All three readiness checks passed with
positive reported GPU residency; all three attempts established cleanup.
This is a failed native acceptance result, not a failure of package installation.

| Attempt | Observed progress | Failure |
| --- | --- | --- |
| 1 | Teaching saved the exact fact and checked its durable job. | Restarted recall announced a retrieval but issued no tool call. |
| 2 | Teaching, restarted recall, correction, and restarted corrected recall passed. | Related forget omitted required source selections and was rejected without commitment. The native forget phase exited 1. |
| 3 | Teaching, restarted recall, correction, and restarted corrected recall passed. | Related forget first supplied an invented JSON object as the digest; a later preview requested an invalid limit. After recovering the actual digest, the model still selected no required sources. These requests were rejected; the native forget phase exited 1. |

No attempt reached a successful deletion receipt, original-source replay, or
final post-forget recall. Attempts 2 and 3 used all eight observed model
responses in their forget phases. Their host exits occurred after 56.896 and
133.592 seconds respectively, inside the 240-second outer deadline. They
were not outer process timeouts. The full raw responses and native output
remain authoritative for the host's continuation and termination behavior.

The service correctly rejected the missing source selections with
`related_forget_source_selection_required` and `committed: false`. The adapter
also rejected the malformed digest and out-of-bounds preview limit. The
frozen erasure and inspection gates remain unchanged; rejection is not
successful forgetting.

## Bindings

- Runtime source: `e419c2aa615e037e639a74b14615889d03f4e47f`.
- Wheel SHA-256: `185f3df7720a32e5ed285d9c507e1e45e7f57370942740b53c4f8178bb97a97e`.
- Committed cohort inputs: `6b20d36739f02d6ab8d6c1a8fbd3e2f312f6b1f2`.
- Host: Hermes revision `14a346345486418aa85ed6526549412ee2665ade`, with the recorded isolated cwd-only compatibility patch.
- Model: `mnemos-native-qwen35-9b:greedy-v1`, digest `243c7dc812e847df8f175763c8e91b6daa88483f141957b76f9e9722e8d526a5`.
- Local Ollama: `http://127.0.0.1:11434`, with Nomic embeddings and all five cognition modules disabled for this host baseline.
- Each readiness record reports 8,044,454,869 bytes of model VRAM residency. This establishes positive reported GPU residency, not that every layer ran on the GPU.
- Started `2026-09-20T05:32:20.592900+00:00`; finished `2026-09-20T05:39:00.173185+00:00`.

The [prospective protocol](../../../plans/2026-09-20-hermes-native-lifecycle-protocol-v8.md)
froze the prompts, three attempts, deadlines, capture, restarts, source replay,
and evaluator before inference. The [import-only preflight](../hermes-native-discovery-preflight-v7/README.md)
discovered eight tools without model calls, and its separate offline report
passed 93 harness cases. The installed package's 1,500 source tests, 648
installed cases, four respective skips, and seven replay assertions remain
[separate evidence](../retrieval-views-checkpoint-v1/README.md).

## Interpretation and next diagnostic

Separating captured evidence from ordinary recall did not establish reliable
native action selection or forgetting in this cohort. Earlier v3-v7 outcomes
retain their original package bindings and failures. These successive
development cohorts are not a controlled estimate of any one change's effect.

The current trace records SDK request metadata and tool descriptions, but not
the full serialized request body or every schema property. A provider-side
tool result recorded before host processing does not alone establish the
exact content the model received. Before changing the model-facing contract,
verify schema forwarding and any host output persistence or pruning. The
missing-source rejection also needs to be distinguished from the model's
confusion between target sources and sources required by additional copies.

Three-host continuity, complete capture, real-model quality acceptance,
Desktop, and the operational pilot remain open. No live host configuration,
source installation, or memory was changed. The archive includes synthetic
fixture evidence selected by the archived helper, not live credentials or
runtime authentication descriptors. `evidence-index.json` binds 212 raw files;
this explanatory README is outside that index.
