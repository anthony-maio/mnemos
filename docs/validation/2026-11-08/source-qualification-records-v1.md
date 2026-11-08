# Source qualification records

Implemented an in-memory development cache of assessed source qualifications. Canonical keys bind the complete candidate revision, retained own sources, designated evidence ID, and exact policy, model, and runtime digests. A changed binding misses the cache. Conflicting assessments cannot overwrite a record under the same frozen identity.

Annotation retains the supplied original candidates and source order, including rejected and unresolved entries. A missing record is unresolved with `cache_hit=false`; an assessed unresolved record is distinguishable. Unavailable results are rejected. The cache never enumerates historical candidates into the result. A caller-supplied live fence runs before and after annotation and remains responsible for scope eligibility, correction, and forget suppressions.

Validation: 28 model-free cases passed, covering candidate and source changes, cross-scope separation, policy/model/runtime changes, own-source binding, omission, post-lookup invalidation, unavailable results, conflicting records, and malformed identity fields. The final module passed mypy with silent imported-module checking. Both changed Python files were formatted with Black; the diff check passed.

These are unit-level control-flow results. The forget test supplies an omitted candidate and a failing revision fence; it does not establish live database or host integration. No inference ran, no semantic result improved, and no production path, persistent cache, live database, or host configuration changed. The next step is the separate qualification provider and query-time selection contract, followed by fresh development inputs and the predeclared evaluation. Original quality, latency, capture, and pilot requirements remain open.
