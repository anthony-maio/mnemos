# Graph v4 development observations

The real Ollama development run at frozen `b7a9d9d` completed five graph cases
and five affective pairs, each with three repetitions. The unchanged authored
dataset used `gemma3:4b` for state classification and `nomic-embed-text:latest`
for frozen embeddings. Exact model, source, dataset, protocol, and vector
hashes are in the [manifest](mechanism-development-v2/manifest.json).

All 15 graph observations passed the recorded insertion-order, independent
activation-energy, paired-snapshot, and scope controls. Cycles were observed
in all 15. Reversing insertion order produced identical edges and retrieval
results. No observation was eligible for associative-recall utility: the
configured graph did not supply the required path to the intended target.
There is no measured graph benefit in this run. Passing these controls also
does not establish successful target-path traversal or edge-cut behavior when
the target has no path.

All 30 affective classifier calls produced parsed states. Fifteen passed the
protocol's distance check against the synthetic intended state. The separate
controlled ranking calculation, using supplied gold states and one frozen
semantic vector per pair, placed the intended memory first in 50 percent of
conditions, versus zero for semantic-only scoring. Top-five difference was
zero. This calculation does not measure end-to-end retrieval using the
classifier's output. The five pairs share one template cluster, so the
reported bootstrap interval supplies little evidence about generalization.

Execution completed and graph correctness passed. The overall mechanism gate
did not pass: this development split contains no held-out observations and
cannot satisfy the 50-case/50-pair held-out coverage requirement. The authored
cases and thresholds remain unchanged for the next run. None of these
observations establishes product utility or human acceptance.

The [full report](mechanism-development-v2/module-mechanism.json), raw provider
outputs, per-observation JSONL files, and failure/coverage report are retained
in `mechanism-development-v2/`. SQLite snapshots, selected inputs, frozen
vectors, and checkpoints remain in `.venv/mechanism-development-work-v2/`.
