# Evaluation queue recovery

The user reported a momentary power loss. The previous coordinator and worker
were absent after a local restart at 18:09 UTC. The interruption snapshot is
preserved separately. The first recovery preflight timed out contacting
Ollama and stopped before changing files.

At 18:29 UTC the same preparation helper verified the restored endpoint's
original model digests, clean frozen d94 source, unchanged coordinator bytes,
and completed-phase artifacts. It copied the old status, lock, and checkpoint
to a separate private recovery directory, then retired only the verified stale
lock. The unchanged coordinator resumed with --resume and --run-performance.
The completed development, preflight, and performance phases were skipped.
This archive records a live coordinator and resumed mechanism worker.

Recovery retained 62 completed case/pair IDs, 112 graph rows, and 150 affective
rows, including partial cases. Counts establish saved execution progress only.
The previous phase's running label remains historical in the append-only phase
list; its old PID was not alive. The newest phase identifies the resumed worker.
No labels, thresholds, source, provider options, or quality outcomes changed.
No held-out quality result was inspected for recovery.

The exact preparation helper ran twice: first the endpoint timeout, then the
successful preparation. Only the latter produced recovery.json. The full
checkpoint copy remains private; its original byte count and digest are in
that report. This archive is recovery evidence, not evaluation completion.
