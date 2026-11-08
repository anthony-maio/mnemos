# Installed Python command startup diagnostic

The unchanged installed capture environment has substantial subprocess startup cost even without capture. Fifty measured observations per case followed five warmups, rotating case order each round.

| Case | Full-process median | Full-process p95 | In-child median | In-child p95 |
| --- | ---: | ---: | ---: | ---: |
| Empty Python body | 78.87 ms | 223.33 ms | 0.0004 ms | 0.0006 ms |
| Hook client import | 147.11 ms | 323.21 ms | 61.79 ms | 68.85 ms |
| Hook client plus entry imports | 147.40 ms | 405.75 ms | 62.89 ms | 69.93 ms |

Full-process time includes subprocess launch, standard output collection, and exit. The inner timer measures only the selected body after interpreter startup and importing the timer. No service, transport, outbox write, or capture ran. All installed package hashes remained unchanged.

This establishes a startup/import component limitation in this observed environment. The empty-command p95 already exceeds 100 ms, while the measured import bodies are comparatively stable. It does not prove a universal latency floor or explain the earlier queued-command p95 in full. Do not subtract p95 values from separate runs to infer worker or persistence cost. The separate importtime sample is diagnostic and is not included among timing observations.

Private evidence under `.venv/queued-command-startup-v1` retains commands, rotated schedule, raw subprocess output, inner and outer observations, dependency-bound package identities, and an importtime trace. The helper is `.venv/check-queued-command-startup-v1.py`. All subprocesses completed; no live configuration changed.

Next verify the current experimental native launcher's artifact and protocol compatibility with queued workers, then measure it at the same complete-command boundary with durable event/job correlation. Python startup cannot be removed from a result while continuing to call that result complete Python command latency. The existing 100 ms failures and original native-host acceptance requirements remain unchanged.
