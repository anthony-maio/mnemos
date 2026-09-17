# Isolated service startup diagnostic

This archive contains one isolated rerun of
`test_autostarted_service_logs_real_retrieval_without_console`. The test
started the service through the real client, performed health, remember,
maintenance, retrieve, and shutdown operations, and recorded the descriptor,
service log, child process, and elapsed time at each boundary.

The run passed one case in 1.707 seconds. Health completed in 952.09 ms. The
remember, maintenance, retrieve, and shutdown calls completed in 7.838 ms,
13.130 ms, 9.109 ms, and 5.031 ms respectively. The service remained owned by
the same child process throughout the recorded calls, and the source hashes
were unchanged before and after the run.

This is a single isolated instrumented rerun at source commit
`6cde46561c0c42c0d6124d86a09d1e2832542c9d`, with the 10 second startup timeout
unchanged. It does not explain the separate full-suite failure where the
initial health result was degraded, and it does not establish reliability,
latency improvement, or a product speedup. The failed full-suite run did not
leave a runtime directory or log preserved for this diagnostic.

The report and JUnit XML are copied byte-for-byte from the executed temporary
fixture. The runner is stored as `.py.txt` so the archive is evidence rather
than an executable test module. No live configuration, credentials, database,
native host, model call, or production network was included.

`evidence-index.json` records exact byte counts and SHA256 hashes for every
archived file except the index itself.
