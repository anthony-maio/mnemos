# Held-out matrix, judged extraction on TypeSafe Jev (v7 corpus)

Run on Colab, one NVIDIA A100-SXM4-80GB, 2026-09-22 19:52 to 21:23 UTC. Source commit `3b3d19d`, bundle `mnemos-heldout-3b3d19d-v7-judged-typesafe`, options `extraction_mode=judged`, `judge_provider=typesafe`, `judge_model=jev-latest` (observed `jev-1.13.0`), `surprisal_mode=novelty`, embeddings `nomic-embed-text` on Ollama 0.34.2. Seven profiles ran as concurrent processes, 200 held-out sessions x 3 repetitions each, 4,200 observations. Wall clock 91 minutes, 37.2M Jev input tokens, $1.56.

This folder keeps the compact parts of the run: `summary.json` (paired bootstrap against baseline), `status/`, `manifests/`, per-profile `measurements/` (gates, sample counts, complete-observation diagnostics), and two derived tables computed from the raw outputs, `judge-usage.json` and `encoding-decisions-by-event.json`. The raw outputs and checkpoints (about 500 MB) stay in the Drive mirror at `D:\GoogleDrive\Root\mnemos-eval\mnemos-heldout-3b3d19d-v7-judged-typesafe\run\`.

## What the switches actually toggle

The `sleep_consolidation` switch controls whether extraction runs at all. With it off (`baseline`, `ablate-sleep_consolidation`) every event is stored verbatim as a fact and no `classify_source` call happens. With it on, judged extraction runs. So every comparison against baseline in `summary.json` is judged extraction versus verbatim storage, not consolidation versus no consolidation. The other four switches were live (judge call purposes in `judge-usage.json` change exactly as each ablation predicts) and produced no measurable difference on this corpus.

## Numbers on complete observations

| profile | supported_precision | critical_retention | answerable_recall@k | no_answer_correct vs baseline | corrections committed |
|---|---|---|---|---|---|
| baseline (verbatim) | 0.285 | 0.529 | 0.385 | | 60/60 |
| full (judged) | 0.552 | 0.524 | 0.372 | -0.042 [-0.073, -0.016] | 0/60 |
| ablate-sleep_consolidation | 0.330 | 0.529 | 0.385 | +0.000 | 60/60 |
| other ablations | 0.52-0.55 | 0.524 | 0.372 | -0.041 to -0.046 | 0-1/60 |

Paired deltas on `answerable_hits`, `critical_retention`, and `critical_retrieval` are exactly zero with zero-width intervals for every profile: the same facts pass and fail the exact-frozen-text test under both paths.

The 60 observations excluded from each judged profile as `partial` are the 20 correction sessions x 3 repetitions. All 60 corrections fail with `missing_accepted_target`.

## Why corrections fail

The correction target is the session's automatic capture, phrased `For <context>, keep this in mind: <fact>.` Under judged classification that event is encoded as `no_durable_facts` in 254 of 300 cases (85%, see `encoding-decisions-by-event.json`). No fact is stored, so `correct` finds no accepted evidence for its target, and the corrected critical fact is never present. The `surprisal_gating` ablation fails the same way, so the novelty gate is not the cause; the `kind` question is. The v12 spike inputs contain no `keep this in mind` phrasing, which is why the spike passed.

Because these sessions are excluded from the paired comparison, the retention tie above overstates judged extraction. Including them would put judged retention below verbatim.

## Open

- Abstention drops 4.2 points under judged extraction while stored facts per session fall from 6.5 to 3.2 and retrieved facts per query rise from 6.1 to 9.8. Not yet explained; diff what `full` retrieves on `no_answer` queries against baseline.
- `supported_precision` and the correction commit rate are not in the notebook's printed summary; the precision gain and the correction failure were only visible in the raw outputs.
- The `keep this in mind:` and `A related decision detail:` prefixes are v7 authoring artifacts. They penalize both paths on exact-text retention and trip the judged `kind` question. A v8 corpus should drop them or score against the sentence after the prefix.
