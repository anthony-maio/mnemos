# Native Hermes v9: verified context, incomplete memory lifecycle

The three scheduled attempts finished zero of three complete lifecycles on
the verified `e419c2a` package. All three readiness checks passed through the
OpenAI-compatible endpoint. All 39 model responses had a matching subsequent
loaded-profile check: the pinned alias/digest, 65,536 context tokens, and
8,044,454,869 reported VRAM bytes. All three attempts established cleanup.

| Attempt | Completed phases | Stopping point |
| --- | --- | --- |
| 1 | Teach | Restarted recall announced retrieval but made no tool call. |
| 2 | Teach, restarted recall, correction, restarted corrected recall | Forget retrieved and inspected the target, received the preview, inspected three related candidates, then stopped without applying deletion. |
| 3 | Teach, restarted recall, correction, restarted corrected recall | Forget inspected the ten candidates and made three apply calls. All returned degraded HTTP 500 with `committed: false`; the host then reported the failure. |

The forget phases exited normally in 34.216 and 94.914 seconds respectively.
They were not outer process timeouts. No attempt reached source replay or
final recall. The HTTP 500 responses require separate service-side diagnosis;
the model's own statement that its arguments were correct is not evidence.

## Request and context evidence

[Boundary analysis](boundary-analysis.json) verifies every recorded request's
hash and byte count and correlates loaded-profile checks with all 39 responses.
For both forget attempts, the complete parsed preview equals the tool result
in the following outgoing HTTP request, correlated through its tool-call ID.
Each preview contains ten candidates and ten required source IDs. The previews
also remain present in the later recorded requests. This establishes the
transported content, not whether the model used it correctly.

The [endpoint diagnostic](../ollama-context-boundary-v1/README.md) established
that the old alias loaded 16,384 through this endpoint despite requesting
65,536. The new alias pins the context in its model definition. Earlier
cohorts retain their original outcomes; their effective context was not
verified after every response. This successor is not a controlled causal
estimate of context size's effect on task completion.

## Bindings and limits

- Runtime: `e419c2aa615e037e639a74b14615889d03f4e47f`.
- Wheel SHA-256: `185f3df7720a32e5ed285d9c507e1e45e7f57370942740b53c4f8178bb97a97e`.
- Runner commit: `c9c61d4cde085d133556cc4daa770d56c47c24db`.
- Model: `mnemos-native-qwen35-9b:greedy-ctx64k-v2`, digest
  `87bc5156d86918e0435431ab0286167a4c79c4a4a259b82fba5f3f75022006e2`.
- Isolated Hermes source: `14a346345486418aa85ed6526549412ee2665ade`,
  with the unchanged cwd-only patch. Imported module origins were checked.
- The live checkout advanced independently to
  `ebe3fcb2f85ebdbe0e469e9d295636ac930246b4`. Its shared interpreter's sorted
  dependency inventory was pinned and checked before and after each attempt;
  no live source, configuration, or memory was changed by this experiment.
- Local Ollama, real Nomic query embeddings, and all five cognition modules
  disabled. The eight tools, memory prompts, erasure requirements, turn limits,
  and deadlines remain those of the frozen v8 tasks.

The [preflight](../hermes-native-discovery-preflight-v9/README.md) preserves
the initial live-revision mismatch and successful successor, plus 118 passing
offline checks. The [protocol](../../../plans/2026-09-20-hermes-native-lifecycle-protocol-v9.md)
was committed before inference. The cohort ran from 06:18:07 to 06:23:31 UTC
on September 20, 2026. The evidence index binds 316 raw artifacts, including
all exact request bodies; this README is outside that index.

Native lifecycle, three-host continuity, complete capture, model-quality
acceptance, Desktop, and the operational pilot remain open. No result here
supersedes the independent T18 ablation results or establishes daily-use
readiness.
