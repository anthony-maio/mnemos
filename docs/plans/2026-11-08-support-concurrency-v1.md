# Optional concurrent support execution

The two support selectors use one immutable snapshot and do not consume each other's decisions. Sequential execution adds their provider wait times. `IntersectionConfig(execution="concurrent")` allows those waits to overlap. The default remains `sequential`; existing evaluations and runtime callers retain their previous behavior.

Both modes preserve candidate order, full-cohort input, evidence selection, snapshot revalidation, and the shared operation timeout. Concurrent results are assembled in selector order rather than completion order. A failure, snapshot change, timeout, or caller cancellation cancels unfinished sibling tasks and waits for their cleanup before returning. Configuration and authorization failures still propagate. Provider outages remain unavailable, never successful abstentions.

This uses cooperative asyncio tasks on the same event loop. It does not move SQLite access or synchronous model inference into threads. Selectors must yield during provider work and honor cancellation. A blocking predictor can still block both tasks and delay timeout handling; cancellation cannot terminate an external provider's computation by itself. Concurrent calls may compete for the same inference server or device, so overlap does not establish lower end-to-end latency.

Model-free tests cover overlap, deterministic result order, sequential/concurrent decision parity, and cleanup after provider/configuration failure, stale evidence, timeout, and caller cancellation. Existing sequential tests remain applicable and unchanged. No model call, ranking evaluation, semantic threshold change, pilot activation, or production default change is part of this patch.

A later latency experiment must first inspect the actual adapters for blocking execution and provider concurrency limits. It must freeze inputs and both execution modes, retain raw component and complete-request timings, and count every failure. Passing this implementation check cannot satisfy the 500 ms complete-retrieval requirement or repair the failed quality gates.
