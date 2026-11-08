# Installed outbox backend command comparison

Status: complete local comparison, failed 100 ms command requirement. SQLite remains an isolated candidate; JSON remains the default. No live activation or publication changed.

Both arms used the same installed v4 wheel from capture candidate 4928d90a and unchanged native v3 launcher. Separate owner processes, memory services, databases and queue directories shared one synthetic project and registry. The frozen schedule used five warmup pairs and fifty measured pairs with alternating order and identical payloads. Command timing includes process launch, request, response collection and exit. Processing and queue drain were checked after each command, outside its timer.

| Backend | Command median | Command p95 | Post-command completion median | Post-command completion p95 |
| --- | ---: | ---: | ---: | ---: |
| JSON | 207.95 ms | 690.03 ms | 189.48 ms | 571.92 ms |
| SQLite | 168.29 ms | 476.37 ms | 220.98 ms | 456.67 ms |

SQLite was faster in 35 of 50 measured command pairs. The median paired SQLite-minus-JSON difference was -35.64 ms. This bounded comparison supports a command-latency benefit in this setup, but neither backend meets the original 100 ms target. Percentiles of command and completion intervals are separate measurements and must not be added to claim a total percentile. Competing services, receipt writes and normal replay activity remain part of the setup.

The independent audit reconstructed all 110 records, matched paired payloads and original event IDs, checked response hashes, reopened both stores, and verified exactly 55 events and 55 completed jobs per arm. Both JSON and SQLite queues were empty; SQLite integrity_check returned ok. Installed files matched the frozen manifest. The runner verified the native binary unchanged. Both owners exited zero, their launcher and service processes were absent, and all four ports were closed.

V1 and its diagnostic probe timed out while project identification waited on Git subprocess output. V2 used DEVNULL stdin and file commands, then stopped after 99 records on PermissionError reading a maintenance command. V3 preserved those runs and added a bounded two-second retry only for that command-file read error. One JSON maintenance command required one retry; SQLite required none. These control reads occur outside the command timer. No capture was retried or sample replaced. The evidence suggests inherited stdin contributed to the initial stall, but does not prove its Windows mechanism.

The verified archive contains 128 members, including the protocol, helpers, raw timings, receipts, owner logs, shutdown records and earlier failure logs. SHA256: e703c52c661dca27a50f25f08b281bc396447abf400b86a95d9fd27cf293cfc3. The adjacent JSON receipt records exact statistics, hashes and cleanup evidence.

This does not establish cold startup, native-host lifecycle acceptance, power-loss durability, 10,000-fact retrieval latency, fresh quality acceptance or actual pilot days. The existing source and installed lifecycle checks remain separate evidence. Further latency work must address the remaining identity-resolution and durable-write costs without weakening scope validation or acknowledgment durability. The original quality gates and daily-use obligations remain open.
