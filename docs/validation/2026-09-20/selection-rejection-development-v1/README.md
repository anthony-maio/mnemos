# Out-of-preview selections receive a clear rejection

Source `5155ebddd8e8cf42df1c67f11a1578f5edfb43ec` replaces two caller-validation
ValueErrors with a typed rejection when a selected related fact or source ID
is absent from the reviewed forget preview. The service returns HTTP 409 with
`related_forget_selection_invalid` and `committed: false`. The fixed safe
message contains no selected IDs. The client accepts only that exact known
response for `forget_related`; other operations and malformed responses
retain their error handling.

The [v11 native failure](../hermes-native-cohort-v11/README.md) motivated this
repair. That attempt copied ten candidate fact IDs, the preview digest, and
the limit correctly, but supplied two source IDs that were absent from the
preview. This change does not replace those IDs, make the invalid request
succeed, or reinterpret the failed attempt.

The archived red run has two expected failures and ten passes before the
production repair. The final focused suite passes 14 cases, and adjacent
related-forget/Hermes coverage passes 58. Tests check the authenticated
service/client/Hermes path, unchanged SQLite state, pending-outbox
preservation, safe error content, valid apply, malformed responses, operation
boundaries, and unchanged HTTP 500 for unexpected internal ValueErrors.
Black passes all five changed files. Strict Python 3.10 mypy passes the four
production files; this is not a strict typing pass for the test module.

Full source/package checks are running against the isolated
`mnemos-source-selection-rejection-v1` checkout. No new installed-package or
native result is claimed. The latest fully verified package remains
`697b672`, with the failed v11 cohort bound to it. The new graph association
diagnostic is development-only and does not change production retrieval.

This development record was created after portable writing packet v22 and
is not included in that snapshot. The packet must be refreshed at final
acceptance. No live host configuration, source, or memory was changed. The
README is outside the raw evidence index.
