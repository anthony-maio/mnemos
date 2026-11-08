# Readable query runner

`benchmarks/local_memory/readable_query_probe_run.py` executes prepared requests serially without grading labels. Before each provider call it writes the complete owner request, exact prompt and schema through the durable journal. It retains the full provider reply before decoding, then records protocol validity, the evidence evaluator's result, provider time and processing time.

A provider exception, timeout or unconfirmed completion stops the run. A completed malformed reply is retained as invalid and is not retried. Failure to write a request prevents dispatch; failure to write a reply or observation prevents the next call. Source fences run before and after the provider boundary and evaluation. Cancellation is recorded and propagated, including when the owner cancels a pending provider coroutine. The 15-second limit is cooperative; it is not proof that an external model process has stopped.

The selected runner, transport, preparation, codec and journal suites passed 36 tests in 1.58 seconds. An added owner-cancellation case and the nine existing runner cases then passed together: 10 tests in 0.73 seconds. These cover 37 distinct cases across both checks. JUnit records are `.venv/readable-runner-v1.xml` and `.venv/readable-runner-cancel-v1.xml`. Black and diff checks passed.

No runtime was launched and no model call was made. The v3 preparation remains unchanged. Before execution, bind this runner and the exact runtime wrapper alongside the prepared input hashes, verify the model and native binaries, and tokenize every full prompt against the configured context/output limit. The existing `EvaluationJournal` provides flush/fsync records and a hash chain; it does not establish hardware power-loss durability or authenticate an untrusted provider.

Provider and processing times exclude initial retrieval, source annotation, host startup and transport outside the provider call. They cannot satisfy the original complete retrieval requirement on their own. The original failed evaluations, source-annotation obligations, quality and latency gates, and actual two-week pilot remain open.
