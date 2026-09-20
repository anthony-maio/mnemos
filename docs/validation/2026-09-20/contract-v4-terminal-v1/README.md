# Contract-v4 real-model matrix: terminal incomplete result

The frozen run finished at 2026-09-20T17:24:00Z on source `3d09658` using
local Ollama, Qwen3.5:9b, Nomic embeddings, and the RTX 4080. It processed all
4,200 planned observations: 4,150 complete and 50 partial. The runner exited
1 with status `incomplete`. This does not pass quality or lifecycle acceptance.

| Profile | Complete | Partial |
| --- | ---: | ---: |
| Baseline | 600 | 0 |
| Full | 589 | 11 |
| Surprisal disabled | 592 | 8 |
| Mutable RAG disabled | 588 | 12 |
| Affective routing disabled | 590 | 10 |
| Sleep consolidation disabled | 600 | 0 |
| Spreading activation disabled | 591 | 9 |

Every partial observation has one `missing_accepted_target` correction error.
In all 50, the first structured extraction response selected
`no_durable_facts`, leaving no accepted fact for the later correction.
The reports contain no other recorded operation errors. These are retained
failures, with no selective retries or changes to the frozen labels or gates.

V4 repairs previously identified scope-label and explicit-body conflicts.
Its inputs are previously observed synthetic scenarios, not an unseen
holdout. The original gates and complete-subset diagnostics remain in the
reports. Exact text matching is a scoring proxy, not semantic factual
precision. Session timing includes supplementary diagnostics and shared
machine load; it is not ordinary recall latency. The baseline report also
preserves the runner's paired comparisons and their stated limits.

`reports.zip` contains all seven report/raw-output pairs and terminal
manifests byte-for-byte. `zip-members.json` binds each member.
`summary.json` records execution counts, original gates, source bindings,
checkpoint hashes, and the partial observations' extraction responses.
The collector checked every report against its original checkpoint and
raw-output file. No memory database is included.

`observed-token-usage.json` separately sums counters returned in provider
response metadata. This posthoc description includes diagnostic calls and
partial observations; it does not replace the original `token_cost` field,
estimate billing, or measure the cost of an ordinary memory query. One
embedding response in the sleep-disabled profile has a null prompt counter
and is counted as missing. The first collector rejected that null under an
overly strict type assumption before creating this archive. Its source is
retained as `collector-attempt1.py.txt`; the corrected collector and verifier
preserve missing counters explicitly.

Verify the archived bytes and independently recompute execution and token
counts with:

```powershell
python verify.py.txt .
```

Archive integrity passes for 4,200 observations, 4,150 complete and 50 partial.
Integrity verification does not change the run's incomplete verdict.
