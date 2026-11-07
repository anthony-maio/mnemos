# Reconciliation reasoning screen v3

Thinking did not improve this screen. It scored 15/20 exact outcomes versus
17/20 for both non-thinking controls, made four wrong-target corrections and
increased median observation latency from 1.4 to 12.4 seconds. It does not
qualify for the planned three-repetition confirmation. Keep the current default.

The [protocol](../../plans/2026-11-07-reconciliation-reasoning-v3.md) and source
were frozen at `06450642ad298386cb1d25fae8f51614dad526e9`. Twenty supplied
classifications and target snapshots ran once under three API/reasoning arms.
All arms received the exact same direct/full prompt and schema, including the
duplicate-identity guard. The chat/off control separates the requested
thinking change from the route change. This is one development screen using
six known scenario clusters, not extraction quality or fresh acceptance.

| Measure | Generate/off | Chat/off | Chat/on |
| --- | ---: | ---: | ---: |
| Exact operation and target outcomes | 17/20 | 17/20 | 15/20 |
| Correct unresolved controls | 6/9 | 6/9 | 5/9 |
| Wrong support or correction target choices | 3 | 3 | 4 |
| Unsupported additions | 0 | 0 | 0 |
| Operational failures | 0 | 0 | 0 |
| Median observation latency | 1,443 ms | 1,405 ms | 12,380 ms |
| p95 latency, nearest rank | 1,816 ms | 1,708 ms | 18,072 ms |

Thinking recovered two controls that chat/off missed: ambiguous environments
and an unspecified numeric setting. It lost four controls that chat/off passed:
an unrelated backup-window target, a different backup attribute, an unrelated
numeric target and a complete prospective requirement without an observed old
rule. The first three became wrong corrections; the requirement became
unresolved instead of being retained as new information. The wrong-environment
case remained a wrong correction.

Several final responses explicitly described the target's mismatched
environment or attribute but still selected `user_revision`, citing correction
intent or shared scope. Those explanations do not justify a match. The original
facts establish the mismatch, and the resulting operations remain wrong. This
also motivates checking the constrained decoder's contribution before making
another semantic prompt change; this run does not establish that contribution.

| Provider accounting | Generate/off | Chat/off | Chat/on |
| --- | ---: | ---: | ---: |
| Prompt tokens | 14,747 | 14,747 | 14,627 |
| Cached prompt tokens | 12,077 | 12,538 | 11,391 |
| Total generated tokens | 1,132 | 1,112 | 14,126 |
| Responses with nonempty thinking text | 0/20 | 0/20 | 20/20 |
| Final text characters | 5,108 | 5,058 | 6,058 |

The installed model advertised thinking support with boolean controls, and the
returned fields confirmed emission only in the enabled arm. All 60 calls
returned one response with a normal stop reason. Thinking text stayed private;
only final-channel decisions were scored. The generated-token count includes
the provider's reported generation work and is not a count of final text alone.
The thinking arm produced 62,627 thinking-text characters.

Every arm used the same existing Qwen3:14b Q4_K_M digest, Ollama 0.40.9,
temperature zero, context 8192 and one request slot. Each had a 4096-token
generation allowance, 3584-token counted input allowance, 90-second provider
deadline and 120-second observation deadline. The maximum native
prompt-plus-schema count was 1,578; the maximum reported prompt and generated
counts were 801 and 1,626. No observation exhausted its output allowance or
failed operationally. These larger generation limits differ from the earlier
studies, while all arms within this run share them.

The independent audit reconstructed all 60 scores from original final
responses and frozen labels. It verified 134 source hashes, all 20 identical
input/prompt/schema groups, request options, route-specific response fields,
thinking emission, token counts and actual native context/slot settings.
All 107 focused tests passed, including 18 new route and budget checks; Black
and strict mypy passed. The optional Python `force_chat` flag preserves
automatic routing by default. The earlier full-suite result keeps its own
source binding.

The [receipt](reasoning-v3-receipt.json) binds 166 verified private archive
members under SHA-256
`3b0096d4d1eedc712298a60b6be262fbcc8d1eab25c4d5fc902f108a6c2fc562`.
The model unloaded, all listed owned processes were absent and both endpoints
closed. The original tree-stop warning remains preserved, with no extra kill
or study rerun. Earlier scores and source archives are unchanged.

The result rules out this reasoning configuration as the next candidate under
the frozen screen. It does not show that reasoning is generally worse, nor
that another model or output contract would behave the same way. A decoder
comparison, full-pipeline and fresh evaluation, persisted extraction and the
actual daily-use pilot remain open. This study changed no live host settings,
installed pilot artifact or publication state.
