# Development delivery checks

This candidate remains unpublished. The explicit restart workflow passes, but the existing release benchmark fails. No release gate or threshold was relaxed, and no PyPI release was made.

## Reproducible checks

The candidate uses Python package version `0.7.0b1` and plugin SemVer `0.7.0-beta.1`. Python 3.12 on Windows was used for local verification. The isolated example uses FastEmbed `BAAI/bge-small-en-v1.5`, SQLite, and unavailable cognition with all experimental modules disabled. It requires a model download on first use and makes no hosted LLM requests.

The example checks explicit storage, durable job completion, a real service process restart, the same fact ID through CLI recall, real MCP stdio initialization and recall/inspection, and an empty result in another project. Loading the Hermes provider entrypoint is an installation check, not native host acceptance.

`scripts/smoke-installed.py` builds its consumer environment outside the source checkout. It checks package/runtime versions, MCP 1.x compatibility, console entrypoints, packaged benchmark data, and the restart example. Historical validation captures are excluded from the source distribution.

## Release benchmark remains failed

Command:

```text
python -m mnemos.benchmark --stores memory,sqlite --retrievers baseline,engine --dataset-pack claim-driving --top-k 1 --repetitions 2 --enforce-production-gate
```

The command exited 2. All six dataset/store comparisons failed in each repetition. MRR lift ranged from -0.4 to 0.0 against the required 0.15. SQLite p95 latency ratios ranged from 22.37 to 43.28 against the configured limit of 4.0. These timings describe this Windows run, not a general hardware-independent performance estimate. The complete output is retained in [release-benchmark.json](release-benchmark.json).

The legacy harness also has interpretation limits:

- Its engine arm inherits the current baseline retrieval profile; it does not explicitly select full-module ranking.
- Its stateful simple embedder is fitted differently in direct-store and engine-ingestion paths.
- SQLite engine recall persists access metadata, while the direct-store baseline only reads.

These are reasons to design a fairer prospective experiment, not reasons to relabel the failed gate. The workflow still enforces it. The separate real-model [recovery and evaluation report](recovery-and-evaluation.md) retains the abstention failure and the original frozen scores.

## Preservation and integration scope

The work is on a separate delivery branch based on `feat/typed-judgment-layer`. Original master commits, untracked plans, the active development worktree, website edits, and evaluation archives were preserved. Live host configurations and the user's memory database were not used by the example.

## Product decision: deferred on 2026-10-28

Further mnemos development and Hermes integration are on hold while Cartograph becomes the first usable Hermes plugin. The explicit restart-and-recall example remains available, and the unpublished implementation and evaluation evidence remain preserved. This decision does not change the failed release gate or authorize a package release.

Reconsider automatic enrichment only after a prospective experiment measures answer support and abstention over the same accepted candidate facts. That experiment is deferred too. Automatic extraction, native host behavior, broader performance, and a daily-use pilot remain open.
