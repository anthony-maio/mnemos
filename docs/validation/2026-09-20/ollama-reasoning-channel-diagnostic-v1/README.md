# Ollama reasoning, API route, and output-budget diagnostic

The local Ollama 0.34.1 server and Qwen3.5:9b model digest are recorded in
each frozen input. This follows the 36 empty-final-response failures in the
[preceding proposal run](../history-reasoning-development-v1/README.md).
No fact, production prompt, live model setting, or acceptance criterion was
changed by these probes. These are previously observed development inputs.

## Results

The route comparison uses three cases: a historical approval report, pure
fiction, and a real writing-project requirement. Each runs through generate
and chat, with thinking enabled and disabled, at a 2,048-token budget.
An interrupted first attempt recorded five responses before its process
disappeared. Its unchanged running status and independent terminal observation
are preserved. A fresh attempt reran all twelve calls with identical inputs;
it was not a selective retry of failures.

| Configuration | Observed result |
| --- | --- |
| Generate, thinking enabled, schema constrained, 2,048 tokens | Three empty final responses; generation stopped after 67-118 tokens. |
| Chat, thinking enabled, schema constrained, 2,048 tokens | Three empty final responses; each exhausted the token budget. |
| Either route, thinking disabled, schema constrained, 2,048 tokens | All six finals validate against the request schema. Both routes still abstain on the historical report. Schema validity does not establish content quality. |
| Either route, thinking enabled, format omitted, 2,048 tokens | All six calls exhaust the budget. Four have no final text; two have incomplete final JSON. None validates. |
| Generate, thinking enabled, schema constrained, 8,192 tokens | The historical-report call still stops after 118 tokens with an empty final. |
| Chat with the schema, or either route without it, thinking enabled, 8,192 tokens | All three historical-report finals validate and retain the report's attribution, date, and explicitly non-current status. Generation uses 3,426-3,427 tokens. |

The larger-budget comparison changes only the budget and request timeout
relative to each matching earlier arm. It tests one previously observed case,
not a representative quality sample. The evidence supports using chat for
explicit structured reasoning and providing enough output budget for a final
answer. It does not establish general extraction quality, reliable latency,
or a universal cause across Ollama versions and models.

## Verification and limits

The collector recomputes every response hash, checks frozen preparation files,
verifies terminal counts, and validates only final-channel JSON against the
original source-bound schema. It uses the existing extraction parser's code
fence handling; it never promotes reasoning text into a final answer or repairs
truncated JSON. Ollama documents this separation in its
[thinking contract](https://docs.ollama.com/capabilities/thinking) and
[chat API](https://docs.ollama.com/api/chat).

The index binds 61 artifacts across four phases, including the interrupted
attempt, requests, complete raw responses, helpers, logs, and status records.
This README is outside the index. A subsequent 36-call development comparison
will exercise the actual frozen SleepDaemon proposal validator through an
experimental chat provider, using all twelve earlier cases and three repeats.
It does not count as complete until its own terminal record is verified.
