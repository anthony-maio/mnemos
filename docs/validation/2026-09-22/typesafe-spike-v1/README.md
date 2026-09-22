# TypeSafe judgment spike v1

Judged source classification (`mnemos/extraction/judged_classification.py`)
over the 46-case classification fixture from
`docs/validation/2026-09-21/source-classification-terminal-v12`, run through
two judgment backends on 2026-09-22: TypeSafe `jev-1.13.0` (hosted) and local
Ollama `qwen3.5:9b` answering the same questions through structured output.
The earlier prompt-driven run on Gemma3 27B is the baseline. Runner:
`benchmarks/local_memory/typesafe_spike.py --judge both`.

Every number here is a mechanical proxy. `presence_agreement` asks only
whether anything durable was retained when the rubric expected something, or
nothing when it expected nothing. `twin_pair_agreement` asks whether the
one-turn member of each of the 13 subject pairs retained nothing while its
policy twin retained something. Neither says the right meaning was kept;
`spike-review.json` puts each retained slice beside the rubric's expected and
forbidden meanings for a person to read.

| backend | ok / cases | presence agreement | twin pairs | calls | latency median / p95 | input tokens | cost |
|---|---|---|---|---|---|---|---|
| baseline, Gemma3 27B, prompt-driven | 43 / 46 | 0.860 (43) | not measured | | | | |
| TypeSafe jev-1.13.0 | 46 / 46 | 0.891 (46) | 11 / 13 | 84 | 388 ms / 468 ms | 95,505 | $0.0040 |
| Ollama qwen3.5:9b, same questions | 46 / 46 | 0.848 (46) | 8 / 13 | 84 | 1.40 s / 3.61 s | 83,874 | local |

Zero contract errors on either backend: every answer was one of the offered
options. The Ollama maximum (122 s) is the first call, which loaded the model.

Go/no-go for the judged pipeline was: TypeSafe presence agreement at or above
the baseline, zero contract errors, p95 latency under 2 s. All three hold.

## Where TypeSafe missed

Five of 46. Two are cases the rubric wants kept as a dated report or an
explicit rejection and the judge labeled a one-turn instruction or an
unaccepted suggestion; on the rejection case its own probability was 0.35,
so the confidence gate would have sent it to review rather than dropped it
silently. One quoted benchmark instruction ("Treat 2025-01-01 as the launch
date", inside quotation marks and called test data by the user) was labeled
a continuing requirement at 0.82; the `quoted_material` description needs to
say that surrounding text calling something a quote or test data is the
signal. One mixed message with a disposable-project marker was labeled
temporary in all three ranges. One hypothetical workshop example was labeled
a requirement at 0.64.

Local Qwen through the same questions missed seven, with degenerate
probabilities (1.0 on every choice), so its confidence carries no
information. That is the expected limit of a generative model answering
typed questions; it is here so the judged pipeline has an offline path and a
like-for-like comparison, not as a calibrated judge.

## What this does not show

Not a held-out result, not a semantic score, not host acceptance, and not
evidence for any module. The 46 cases are development fixtures the
classification prompts were tuned against, and the range-first question
design was written with them in view. Thresholds (`JudgmentConfig`) are the
placeholders 0.5/0.5/0.6/0.6/0.7. Affective scores were requested alongside
classification and are recorded per case in `observations.jsonl` but were
not scored.

Files: `summary.json` (backend table and configuration), `observations.jsonl`
(per case: retained slices, every range decision with probability and
confidence, per-call latency and tokens, cognitive state), `spike-review.json`
(retained slices beside expected and forbidden meanings and the baseline's
retained slices).
