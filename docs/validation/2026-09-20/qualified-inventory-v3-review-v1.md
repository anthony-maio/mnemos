# V3 contract repair and offline semantic review

Candidate 5134557 adds separately named v3 APIs. An optional retention-request
quotation is allowed on an ordinary assertion, but must still be nonblank and
present in that same source. Explicitly retained uncertainty still requires a
request quotation. The original v2 functions and tests preserve their original
restriction. Scope, retained kinds, source binding, qualifications, write policy
and rendered length checks remain unchanged. Combined v1/v2/v3 tests pass 51
cases; formatting and Python 3.10 mypy pass.

The [offline replay](qualified-inventory-v3-offline-replay-v1/README.md) uses the
exact 46 stored v2 response texts with zero new model calls. All 41 originally
valid inventories and retained outputs remain identical. Four previously
rejected responses now pass the runtime contract; one still fails. The original
v2 experiment still has five errors. This replay is a separate result.

The four newly accepted cases have different semantic outcomes:

- dated-report-06-policy retains the ISO report date requirement, observed retry
  count requirement and future-incident scope together.
- benchmark-date-10-policy retains the fixture-versus-project-fact boundary and
  explicit user-confirmation exception, but splits the future requirement into
  an ambiguous standalone "Keep this boundary" item. Attachment remains flawed.
- mixed-marker-13-policy retains the marker and continuing outage-test policy
  as separate, meaningful assertions, including external-tool and checked-receipt
  constraints. Its request quotation preserves the explicit future-test wording.
- P01 retains concise, decision-first plans and the short-paragraph preference.
  It repeats that preference both as a qualification and a separate assertion.
  The preference sentence in retention_request_quote is not itself an explicit
  request to remember. Optional quote validity does not establish that intent.

P06 remains rejected because its qualification begins "Telemetry" while the
source uses lowercase "telemetry" in that clause. Exact quotation checking was
not relaxed. Its rejected content also says telemetry is not enabled, converting
a requirement into an unsupported observation. That semantic defect is visible
in the original response, but no accepted memory is produced for this case.

The previous N02 speculation, mixed-marker-13-one-turn contamination and lost
benchmark-quote-04-policy remain unchanged. No semantic acceptance threshold was
changed, no aggregate accuracy is reported, and this candidate is not adopted.
Reconciliation and durable service behavior remain outside this experiment.
