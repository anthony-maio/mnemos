# Evidence-index installed validation

Product `f5a944ea` passes isolated installation checks using its newly built
wheel. Explicit remember, service restart, same-ID recall, MCP inspection and
project isolation pass with real BGE embeddings. All 96 package files match the
wheel, all 85 Python files match committed source, and all 57 dependency versions
match the prior installation. The 17 cached model assets remain unchanged.

An additional installed check confirms that evidence lookup uses the new
`(fact_id, created_at)` index without a temporary sort. Equal-time evidence keeps
its insertion order after reopening the store. Database integrity and foreign
key checks pass. Generated Codex guidance retains its reviewed hash.

The [receipt](fact-evidence-index-install-v1-receipt.json) and
[smoke result](installed-smoke-f5a944ea.json) bind these results to the artifact.
Wheel SHA-256: `07c5b424bc7731cacbebb45f44709b3f19509c2cde3150a6451c92d12b27e6ae`.
Private archive SHA-256: `233fe5efc1e81e70a47c6185ebd1c0ee065b29045057a0df6ee2ec289a08516b`.

The subsequent [full regression](full-regression-fact-evidence-index-v1.md) and
[complete installed comparison](fact-evidence-index-performance-v1.md) are
recorded separately.
Earlier native-host acceptance remains bound to its original source. This check
does not activate the pilot or change live host configuration. Nothing was
published.
