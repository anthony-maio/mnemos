# Contrastive answer-support comparison v6

Status: prospective development comparison. Commit all inputs before inference,
use one fresh output directory and preserve every attempted observation.

The v5 baseline's hybrid errors were unresolved retention conflicts, a different
attribute, an unsupported historical extrapolation and an unknown future state.
The three-check candidate added errors and latency. Test `contrast_v5`: one
batch containing a choice for every eligible candidate, with compact rejection
options for a different request, an unestablished period, an unsupported source
or an unresolved relevant conflict. Each positive choice still identifies one
of that candidate's own retained sources. The model cannot add or rewrite text.

Compare against unchanged candidate-at-a-time `joint_v1`. This changes wording,
shared comparison context and batching together; it does not isolate a single
cause. Both receive the same explicit reference date, November 6, 2026, and
identical retrieved IDs, versions, retained evidence and accepted-fact snapshots.
Comparison dependencies invalidate the whole candidate result if any supplied
fact, source or eligibility changes. Production retrieval remains unchanged.

Reuse the two frozen development datasets from the
[v5 protocol](2026-11-06-answer-support-checked-v5.md): 22 scenarios, 44 questions,
27 answerable and 17 without answers. Preserve their hashes, expected sets and
all four retrieval arms. Run one repetition: 176 frozen retrieval inputs and
352 treatment observations. These are 44 questions, not 352 independent samples.
No extraction runs and no fresh-evaluation claim follows.

Keep Qwen3 14B Q4_K_M digest
`bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`,
Ollama 0.40.9 executable SHA-256
`0957bdaa7b52e163cdc7feb1275653f27c6b32e8ba309e957bdbb83493e97a4b`,
and BGE-small English v1.5 embeddings. Use private loopback port 11437 with
one loaded model and one parallel request. Keep context 8,192, temperature
zero, thinking off, output limit 1,024 tokens, operation deadline 15 seconds,
typed request cap 64,000 bytes and rendered prompt cap 5,120 bytes. Require an
offline measurement of actual guarded prompt construction before model startup.
Preserve the top-five, 1,200 UTF-8-byte retrieval bound, input shuffle seed
20261106 and treatment-order seed prefix `temporal-v4:`.

Require complete probability distributions, unit sum within 0.01 and a selected
option with maximal probability. Retain the 0.8 selection threshold. These
model-reported probabilities are not calibrated. Failed operations receive no
abstention credit. A compact shared request changes request and token counts;
measure those counts and latency instead of presuming improvement.

Warm-load with an empty prompt. Record device occupancy and original prompts,
schemas, responses and outcomes. Independently verify source hashes, all 176
pairs, unchanged accepted snapshots, grading and request/response accounting.
Report hit and clean-hit counts, abstentions, unsupported facts, failures,
latency and usage for each treatment, arm and dataset. Keep negative results.
Unload the model and stop only this private runtime. No cloud fallback,
download, live pilot activation or publication is part of this experiment.

The 90% recall and abstention targets remain development decision criteria.
Passing would still require a frozen candidate followed by separate fresh
evaluation, extraction evaluation, context-efficiency measurements and actual
pilot use. All older protocols and scores retain their original bindings.

```powershell
.venv/Scripts/python.exe -m benchmarks.local_memory.support_temporal_comparison `
  --candidate-protocol contrast_v5 `
  --dataset benchmarks/local_memory/fixtures/answer-support-development-v1.json `
  --dataset benchmarks/local_memory/fixtures/answer-support-temporal-development-v1.json `
  --output .venv/answer-support-contrast-v6 `
  --endpoint http://127.0.0.1:11437 `
  --source-root . `
  --protocol docs/plans/2026-11-07-answer-support-contrast-v6.md `
  --embedding-cache .venv/support-runtime/embedding-cache
```
