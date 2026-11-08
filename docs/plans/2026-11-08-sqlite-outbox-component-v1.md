# SQLite outbox component comparison v1

Freeze candidate source, helper and this protocol before execution. Compare the existing atomic JSON outbox with the isolated SQLite WAL outbox using synchronous=FULL. No adapter, installed wheel or live pilot selects the new backend. Use separate new private files on the same volume, deterministic synthetic payloads, five warmups then fifty alternating pairs. Preserve all samples and failures.

Time append through its returned receipt and acknowledgment through its returned result separately. Both arms start each sample with an empty queue; SQLite retains its connection across samples. Do not hide SQLite acknowledgment/commit cost by reporting append alone. Compare stable IDs and exact pending payloads after every pair, acknowledge both, then verify both queues empty. After all samples, close/reopen and verify emptiness. Confirm WAL and FULL remain selected, and hashes of candidate source stay unchanged.

Report minimum, median, nearest-rank p95 and maximum for append, acknowledgment and their per-sample sum. Setup, reads, validation and reopen are outside timed intervals. Cold initialization is included only in warmups and is not a warm-latency result. This is a source-level persistence experiment, not complete hook command, native-host, installed-package, scale or power-loss validation. The complete 100 ms requirement remains unchanged. No optimization may weaken commit durability, firewall behavior, original event identity, conflict rejection, concurrent-client safety or forgetting/replay semantics.

Keep all receipts and raw timings. A favorable component result only justifies further installed lifecycle and command measurement; it is not promotion. A failed result must remain visible.
