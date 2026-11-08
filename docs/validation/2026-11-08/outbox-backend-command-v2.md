# Outbox command comparison: incomplete

The v2 comparison stopped after 99 recorded captures. It cannot establish either backend's timing acceptance or a complete paired comparison. All samples and earlier failed runs remain preserved. The installed v4 package and native v3 executable were unchanged.

The original v1 run timed out on the first JSON warmup. A separate bounded probe also timed out; its stack traces showed project identification waiting in Git subprocess output collection while the owner main thread read stdin. Both runs shut down their owners. These observations identify the blocked call, but do not prove why Git's output reader stalled.

V2 replaced stdin control with sequential command files and gave the owner DEVNULL stdin. It reached pair 49, then the JSON owner received PermissionError while opening command-49.json for a maintenance request. The owner finally block stopped both its services. The driver stopped the SQLite owner. A separate check verified both owner and launcher process IDs absent, all four ports closed, and both stopped records present. See the adjacent JSON receipt for evidence hashes and cleanup checks.

The file control failure occurred outside the capture command timer. Nevertheless, the prescribed 55 pairs were not completed, so no acceptance percentile is reported. Removing inherited stdin allowed substantial progress, but the control mechanism also changed; this is evidence for further diagnosis, not a proven Windows stdin root cause.

Next: make control-file reads tolerate bounded transient sharing failures, retain failure telemetry, and ensure cleanup verification also runs for owners that exit before the driver's finally block. Freeze a new protocol and fresh output directory before another full comparison. Do not replace failed samples or alter the installed product to hide harness failures.

No live pilot files, memory databases, canonical delivery or publication state changed. The original quality, timing, fresh acceptance and daily-use requirements remain open.
