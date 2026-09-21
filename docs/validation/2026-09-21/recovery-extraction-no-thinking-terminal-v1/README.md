# Recovery extraction control: valid responses, marker omitted

Both saved-event calls returned parseable no_durable_facts JSON with think=False, unlike the earlier empty structured responses. The assistant acknowledgment appropriately emits no fact. The user message contains an independent project marker alongside temporary instructions; omitting that marker fails the unchanged extraction criterion. No store was opened and no facts were written.

The requests were reconstructed from original d937892 wheel code and saved events. Thinking false also changes chat to generate. This is a single reused diagnostic, not isolated reasoning causality, lifecycle recovery, semantic acceptance or a latency benchmark. Total run time includes model loading and any provider retry behavior; final-response counters do not measure all attempted work.
