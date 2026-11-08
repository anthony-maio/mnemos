# Revalidate authorization before returning retrieval results

Canonical commit `25126632` repairs a reproduced retrieval authorization race. The service checked scope before awaiting the engine but returned its results even if the project was disabled during that await. Four regression cases reproduced the failure across memory/captured-evidence views and full/compact responses.

The test uses a deterministic engine stub which disables the project in the disposable registry and returns a scoped chunk. It tests the service response boundary, not provider correctness or retrieval ranking. Before the repair all four cases returned successfully rather than rejecting access.

LocalMemoryService now refreshes file-owned authorization at direct retrieval entry and again after the engine await, before collecting evidence or rendering either response view. Both checks validate the request context. Eight regression cases cover disablement before the direct call and during the await across the two views and two response formats. The pre-call cases additionally verify that the engine is never called.

Validation: 54 targeted registry, retrieval-view, compact-response, concurrent-revision and memory-service tests passed in 14.09 seconds. The four original failing cases are retained in `.venv/retrieve-revoke-red-v1.xml`; passing results are in `.venv/retrieve-revoke-green-v1.xml`. Black formatting and `git diff --check` passed.

This prevents the demonstrated service response after revocation. It does not retract work already sent to a provider, reverse engine-internal retrieval side effects, or make external-file edits atomic with final network transmission. No live configuration or installation changed. This source commit has not been built into a newly verified package, and previous installed-package/timing receipts remain bound to `d9ab39ba`.
