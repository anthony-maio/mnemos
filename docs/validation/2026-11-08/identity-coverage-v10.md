# Independent identity coverage v10

Status: measured and independently audited; rejected. Local development evidence only. No promotion, publication, live configuration change or original pilot day.

The extra coverage check made reconciliation substantially worse. It produced 42/72 correct operations and 73/198 correct relations, with two unsafe target choices. The frozen gate required all operations and relations correct, no unsafe acceptance and no operational failures. It failed. Do not run the original-screen or three-repetition confirmation for this candidate.

| Measure | Historical v8 | Identity coverage v10 |
| --- | ---: | ---: |
| Correct operations | 66/72 | 42/72 |
| Correct relations | 167/198 | 73/198 |
| Both correct in a view | 53/72 | 20/72 |
| Correct reviews | 24/30 | 28/30 |
| Wrong target choices | 6 | 2 |
| Wrong additions | 0 | 0 |
| Operational failures | 0 | 0 |

The 72 views represent twelve development seeds, not 72 independent fresh scenarios. V8 is a historical control, not an interleaved run. Every v10 baseline response was byte-identical to its corresponding v8 response. Within v10, the coverage transformation repaired six relations and regressed 100: 94 correct different-variable judgments became uncertain, and six correct same-variable judgments were rejected as incompatible. These counts isolate the recorded transformation; they do not establish performance on fresh data.

The candidate withheld 18 previously valid corrections and ten valid additions. It returned 56 reviews, fourteen corrections and two additions; only twelve corrections and 28 reviews were correct. Unrelated distractors were often labeled unspecified, which made the final ambiguity check refuse valid changes. The same missing-environment requirement still selected production without source authority in its base and renamed views, while the coverage response labeled both production and staging covered. Fewer wrong corrections came with a large loss of useful behavior and did not eliminate unsafe acceptance.

All 396 planned requests completed with normal stops: 198 baseline comparisons and 198 coverage checks. Coverage returned unspecified 106 times, covered 72 times and incompatible twenty times. Across twelve seeds, operations were stable in seven, relations in eleven and both in six. Only five seeds had all operations correct, and one had every operation and relation correct. One of 34 literal pairs produced differing final relations across views.

Median case latency was 7504.28 ms; p95 was 11335.20 ms. These are development reconciliation timings, not complete capture or retrieval measurements. Native input counts peaked at 1572 tokens. Provider receipts recorded 286352 prompt tokens, 47015 cached tokens and 10571 generated tokens. The maximum output was 52 tokens. A separate structured smoke response was excluded from quality scoring.

The offline audit verified all 166 bound source hashes, fixture and protocol hashes, recorded artifacts, every request against the frozen preflight, native token counts, both raw response layers, all 396 value-span fields and all 72 runtime replays. It independently reconstructed relation transformations, operation scores and seed stability without making model calls. The prior preparation recorded 97 focused tests, including fifteen new tests, and strict type checks across four modules; those are implementation checks, not semantic acceptance.

The private archive contains 201 members and passed archive and member-hash verification. Native logs confirmed all 397 sampling records, including the smoke. The measurement exited successfully; the wrapper unloaded the model, stopped its owned server and verified ports 11437 and 64395 closed. An independent process and port check confirmed cleanup. The server exit code after intentional termination was 1; the tree-stop command returned 0.

Evidence: [frozen protocol](../../plans/2026-11-08-identity-coverage-v10.md), [preparation](identity-coverage-v10-preparation.md), and [receipt](identity-coverage-v10-receipt.json). Raw prompts, responses, inputs, runtime logs, replay audit and source snapshots remain in the private local archive, whose SHA256 is recorded in the receipt.

Keep this path experimental for reproducibility. Another broad identity prompt is not justified by this result. Before further inference, review whether a narrower ambiguity mechanism can preserve explicit mismatches, distinguish setting values from identity conditions, and handle a single qualified target without inventing omitted source detail. Any design must cover the remaining relation errors and receive a new frozen protocol. The earlier quality, complete latency, fresh acceptance and actual daily-use obligations remain open. The optional explicit-memory pilot remains separate and unchanged.
