# Live tokenizer retrieval observation harness

The observation harness now opens a separate verified SQLite copy for each
retrieval and injects the tokenizer through the engine's existing constructor.
It uses the actual local service, baseline profile, fixed scope, top five and
the specified context budget. Timing covers the service retrieval call; database
copying, integrity checks and subsequent audit work are outside that interval.

Every observation retains the service response and candidate traces, including
budget omissions. It checks the original snapshot hash, accepted fact/source/job
projection, SQLite integrity and foreign keys. The original file must keep its
hash after the observation. An existing destination is refused, and the copied
store is closed on success or failure. Experimental modules must be disabled.

Three integration checks pass in 3.17 seconds. They verify that a supplied
counter's exact count and identity reach the service response with retained
source references, a failing callback matches ordinary byte fallback including
the returned facts, and an existing destination remains byte-for-byte unchanged.
These tests use a deterministic embedding provider and synthetic counter; they
are not the scheduled real-BGE/live-Qwen measurement.

The first scoped type check rejected `hashlib.file_digest` under the project's
Python 3.10 target. Incremental SHA-256 reads replaced it. The three checks pass
again after that repair, scoped mypy passes with `--follow-imports=silent`, Black
passes for both files and `git diff --check` passes.

The benchmark harness and tests are committed separately from product code.
The executable schedule, runtime ownership controller, canonical source and
dependency freeze, all 408 observations and independent audit remain pending.
No native tokenizer runtime, model generation, live configuration change or
original-database write occurred in this increment.
