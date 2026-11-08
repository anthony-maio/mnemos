# Accidental recency matching in the legacy candidate

Fresh fixed-snapshot checks reproduced an unintended ranking change in the legacy engine. Its substring search found `now` inside `known` and `unknown`, and `current` inside `concurrent`. Those non-temporal queries consequently enabled recency reranking and displaced the stronger semantic candidate.

The reproduction used identical stored facts, fixed two-dimensional embeddings, a fixed UTC engine clock and explicit creation/activity timestamps. Both memory and SQLite backends were tested with zero and 1,000 prior accesses. The old implementation failed all twelve non-temporal cases while passing twenty explicit temporal controls. The failing record remains at `.venv/recency-word-red-v1.xml`.

Isolated commit `098ad81d` on `wt/mnemos-recency-word-boundary` replaces substring matching with word-boundary matching. It retains current, now, latest, today and recent, including the intended currently/recently forms. The change does not modify scoring weights, source facts, timestamps, embeddings, candidate visibility, scope, thresholds or any frozen benchmark.

All 110 targeted tests passed in 2.10 seconds, retained at `.venv/recency-word-green-v1.xml`. This includes forty new combinations across both stores and access histories, plus existing engine and benchmark tests. The checks also confirm that retrieval does not change content-recency timestamps. Black and Git whitespace checks passed. The candidate and changelog are committed locally; no package was built or installed for this change.

These fixtures diagnose a lexical trigger, not answer-support accuracy. The temporal heuristic still does not interpret negation, multiple dates or whether source evidence proves current validity. This finding cannot retroactively explain the closed prerelease benchmark, whose missing observations remain missing. It does not turn that failed gate into a pass.

The fix is confined to the legacy candidate, which is separate from the newer canonical service implementation. Live configurations, databases, original evaluation evidence and pilot settings are unchanged. Original quality, complete latency, fresh acceptance, actual two-week pilot and publication requirements remain open.
