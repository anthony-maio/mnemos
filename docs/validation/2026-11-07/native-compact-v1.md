# Native Codex compact recall screen v1

Both formats answered all eight fields correctly in all three sessions. Compact
replies contained the same five facts and complete context, with 58.67% fewer MCP
text bytes. The native traces confirm that the complete context reached the model
in all six sessions, including the conflict and unknown-value qualifications.

This is a small synthetic screen of the source checkout. It does not establish
fresh retrieval quality, a general token-saving percentage, or pilot readiness.
The installed package and live host configuration were not changed.

The fixture, runner, grader and [protocol](../../plans/2026-11-07-native-compact-v1.md)
were frozen at `670e4b36` before inference. The run used Codex
`0.164.0-alpha.11.1`, `gpt-6-astra`, high reasoning, six fresh disposable profiles,
and one model-issued `mnemos_retrieve` call per session. Only the requested
response format differed between prompt templates. No capture hooks or local
cognition model ran. The fact and source-evidence snapshot remained unchanged;
retrieval access metadata was allowed to advance.

| Session | Format | Correct fields | Input tokens | Cached input | Output tokens | Native seconds | Inference requests |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | full | 8/8 | 70,711 | 54,144 | 232 | 17.469 | 4 |
| 2 | compact | 8/8 | 52,407 | 41,472 | 205 | 13.297 | 3 |
| 3 | compact | 8/8 | 69,384 | 58,240 | 237 | 15.047 | 4 |
| 4 | full | 8/8 | 70,692 | 58,240 | 222 | 16.797 | 4 |
| 5 | full | 8/8 | 73,887 | 50,432 | 204 | 18.657 | 4 |
| 6 | compact | 8/8 | 70,723 | 58,240 | 204 | 15.234 | 4 |

These are actual terminal usage counters for the whole native turn. They include
host instructions, tool discovery, schemas and repeated input. Cached input is a
subset of total input. The retained native objects also reported zero cache-write
and reasoning-output tokens. These counters are not an arbitrary-text tokenizer
or a billing estimate.

Full replies were 9,093 bytes each; compact replies were 3,758 bytes each. In the
first four sessions, Code Mode printed the text content alone. Including its
status text, the retrieval output shown to the model was 9,140 bytes for full and
3,805 for compact. In the last two sessions, Code Mode printed the entire MCP
object, including repeated structured content: 19,940 and 8,242 bytes respectively.
The audit matched these outputs to subsequent native inference requests, rather
than assuming the raw MCP reply was the model input.

Compact used fewer total input tokens in each planned pair: 18,304, 1,308 and
3,164 fewer. The first pair also had one fewer discovery/inference step, which
accounts for much of the large difference. Across the three sessions per format,
total input was 215,290 for full and 192,514 for compact, a descriptive 10.58%
difference. Cache behavior and model-generated tool code also varied. Do not
attribute that percentage solely to compact serialization or extrapolate it to
other workloads. Median native durations were 17.469 and 15.047 seconds; this
sample does not establish a general latency improvement.

All six native sessions completed without retries or transport failures. The
independent audit re-read the original native streams and correlated model calls,
code cells and MCP calls. It checked exact arguments, fact IDs and versions,
scope, assertion status, literal context, foreign-project exclusion, typed answer
fields, usage counters and 85 bound source/protocol/helper files.

Two setup preflights stopped before inference because the seed writes had been
queued without maintenance. The corrected preflight processed those writes and
confirmed all five facts were available within the context budget. These attempts
are retained separately and are not model observations. The first offline audit
also stopped because it compared three selected usage fields with the complete
native usage object. The native object had two additional zero-valued fields.
The corrected audit checks the original object separately and then compares the
selected fields. No model request was repeated or response changed.

Validation: 110 focused tests, including 18 new grading/usage checks, and 12
documentation checks passed. Black passed for all three new Python files; strict
mypy passed for the fixture/grader. This turn made no production-code changes.

The six recorded native process IDs and any matching new MCP children were absent
after cleanup. Temporary authentication copies were removed, the source
credential file was unchanged, and the owned service endpoint was closed. The
private pilot still has no database or runtime and has accumulated zero days.

The private archive has 338 members, with every member hash verified, and SHA-256
`879c5564f7293e926d295702ef895e49111b826f884291dcfa1027548e2fdcbf`.
The [aggregate receipt](native-compact-v1-receipt.json) contains the measurements
and bindings. Raw native sessions, inference payloads and synthetic SQLite data
remain private. Authentication files and runtime credentials are excluded.

The earlier [transport measurement](compact-v1.md) measured additional inspection
cost. This native screen used recall alone. It does not claim that compact plus
inspection is smaller, that automatic extraction is reliable, or that the
remaining fresh evaluation and daily-use pilot requirements have passed.
