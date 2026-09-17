# Combined checkpoint failure

Frozen source `9aba209` completed the full suite with 856 passed, four skipped, and one failure in 991.85 seconds. Formatting, type checking, JavaScript syntax, build, and Twine were not run by this checkpoint after pytest failed.

The failing regression was `test_measure_retrieval_performs_one_metadata_update_per_engine_query`. The benchmark called `embed_document` directly on a legacy provider exposing only `embed`, causing AttributeError before retrieval. An isolated run reproduced the same failure. The proposed repair uses the shared role-dispatch helper; the expanded regression preserves the original access-write assertions and checks exact unprefixed and query/document-prefixed payloads. The subsequent focused run passed 51 tests, but this failed full result remains unchanged and is not a passing package checkpoint.

The repair is committed in `17c55ef`. `repair-focused-pytest.xml` preserves the separate 51-test passing run; it does not replace the original failed suite.
