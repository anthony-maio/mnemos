# Native versus Python queued command timing

Use the verified installed queued worker wheel `f9160ee1ca1bc09f169c09f7ad3c61ca5e45019c66150b29d14903fbebb3f5c3` and native launcher `ec52c79a0ae613ae336836b3e82ddf853a601a0de3f7cd3aa0047318d06ee12b`. Compare the native executable with the installed Python `-I -m mnemos.hook_client codex Stop` command. Both target the same queued resident worker and real memory-service owner.

Record authenticated worker receipts through the same worker-side observer for both arms. Its trace overhead is included in command timing; neither arm gets a separate client trace wrapper. Preserve valid host stdout and require a correlated queued receipt, original event identity, persisted event, and completed job for each command. Completion work and queue drain occur before the next sample and outside command timing.

Run five warmups per arm, then 50 pairs with alternating arm order. Time process creation through complete stdout/stderr collection and exit; report nearest-rank p95 over measured samples only. Unique synthetic event identities avoid deduplication across samples. Freeze helpers, complete commands, schedule, package and native identities, and dependencies before execution. Preserve every failure; do not selectively repeat a failed sample.

This is an initially empty synthetic store with deterministic embeddings and no cognition, not a 10,000-fact recall run or native Codex host test. Both workers and service are warm. Fixed trace instrumentation and environment conditions constrain comparison with earlier runs. Keep the 100 ms command target and original native-host/pilot gates unchanged; a passing component study cannot promote the candidate by itself.
