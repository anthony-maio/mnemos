# Benchmark query evidence

The delivery refresh verified public master at `866444378e5a42598bba0548f6460d07adcded10`. [PR #5](https://github.com/anthony-maio/mnemos/pull/5) is merged. [PR #7](https://github.com/anthony-maio/mnemos/pull/7) remains open at `9c9ebf8406856057aac051862cb6f7bf55a802d8`. Neither observation publishes the local transaction repair or prerelease artifact.

The local 0.6.1rc1 release benchmark remains failed: eleven of twelve comparisons passed, with one failed in-memory contradiction comparison. Its saved report lacks returned chunk IDs and content chronology. The existing packaging repair remains a separate unpublished candidate.

New isolated commit `3868680e` adds versioned per-query evidence on top of that packaging candidate. Reports retain dataset-to-chunk mappings, pre-query stored metadata, ordered returned IDs, relevant IDs, query hashes, scope inputs, UTC start times and copied chunk timestamps/version/access counts. The saved IDs and latencies are exactly those used for aggregate metrics. No ranking code, fixture, threshold or frozen result changed.

All 21 targeted benchmark tests passed in 2.12 seconds, retained at `.venv/benchmark-evidence-v1.xml`. New tests reconstruct metrics from serialized evidence across both stores and retrieval modes, retain repeated query observations, and verify later chunk mutation cannot rewrite previously copied timestamps. Black checks pass for both Python files. Targeted mypy passes for the changed source module with imported-module diagnostics suppressed and Python 3.12 selected. The initial Python 3.10-target check stopped while parsing newer NumPy stubs in the validation environment; it did not establish a code failure or Python 3.10 typing success.

The new pre-query snapshot can warm storage, and serialization adds work between queries despite being outside the query timers. The documentation states this explicitly. No latency equivalence, fresh quality result, supported-platform matrix or release readiness is claimed. The new candidate is committed locally, clean and uninstalled; canonical delivery, live configuration and pilot state are unchanged.

The next release investigation can use this evidence format on a separately specified fresh experiment. It cannot recover the missing observations in the failed run or justify replacing that result with a retry. All original quality, complete latency, actual pilot and publication obligations remain open.
