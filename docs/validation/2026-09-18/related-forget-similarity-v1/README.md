# Reviewed similarity candidates for forgetting

Source based on `e1ffc65c78d5a779320a0579756a4384fbdc3f9c` plus `source.patch` passed 1,022 tests with 4 skips, Black, strict mypy, and JavaScript syntax. The report binds all Python source/test inputs, UI JavaScript, and project configuration before and after validation. Temporary fixtures were outside Git on C:.

Ten initial regressions failed against the previous exact-only preview. The first focused run then passed 45 tests. The final full suite includes 13 similarity regressions covering explicit selection, read-only discovery, replay suppression and reopen, scope filtering, deterministic bounds, stale embeddings/content/version/dependencies, missing vectors, shared sources, and redacted foreign sources. Vectors are controlled fixtures; these tests do not measure embedding quality.

Exact current/historical assertion matches are supplemented by at most 20 positive-cosine nearest captured facts using the current target and candidate stored document vectors. Scope and source kind are filtered before ranking. Suggestions are advisory and require explicit selection. Apply regenerates the plan inside the existing transaction and binds embedding hashes, scores, versions, content, and source/job dependencies to its digest.

`complete` describes the returned selection and dependency plan, not exhaustive discovery. Raw episodic paraphrases without indexed facts, missing vectors, historical-only meanings, and candidates outside the similarity bound may be missed. Exact-match overflow or dependency overflow still prevents applying an incomplete plan.

Native v6 remains a recorded failure. No native host, installed wheel, or real-model quality acceptance was established by this source run. The frozen Ollama evaluation uses an earlier source and was not changed.
