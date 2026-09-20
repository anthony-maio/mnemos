# Durability comparison v1: completed semantic review

All 52 calls completed: 26 runtime-valid responses per arm, zero runtime errors.
This review covers every case pair against the frozen rubric and ambiguity
notes. It is non-blinded descriptive review of reused development cases, not
an independent accuracy estimate. No facts were committed by the experiment.

| Pair | Observed outcome |
| --- | --- |
| reply-format-01 | Both abstain on the temporary exact reply and retain the scoped release policy. |
| tool-use-02 | Production retains the one-answer tool restriction; candidate abstains. Both retain the scoped ongoing policy. |
| outage-03 | Production generalizes the current incident procedure; candidate abstains. Narrow incident context is an ambiguity. Candidate loses the explicit revocation condition in the continuing policy. |
| benchmark-quote-04 | Both abstain on the quoted command and retain the ongoing evidence policy. |
| benchmark-untrusted-05 | Both abstain on the temporary fixture instruction and retain the benchmark-scoped policy. |
| dated-report-06 | Both abstain on the one-report context, permitted by its frozen ambiguity note; both retain the future report-format requirement. |
| rejected-proposal-07 | Both preserve the rejected proposal without adopting it. Candidate policy retains rejection status but omits the requested preservation of useful rejection history. |
| reply-style-08 | Production retains a next-reply preference; candidate abstains. Both retain the ongoing report-style policy. |
| outage-command-09 | Both abstain on the one-incident instruction, permitted by the ambiguity note, and retain the standing retry policy. |
| benchmark-date-10 | Both retain a prohibition derived from the current quoted fixture, violating the no-durable-fact expectation without adopting the false date. Both retain the ongoing evidence policy. |
| temporary-plan-11 | Both abstain on the unapproved hypothetical and retain the explicit no-approved-fallback state and approval requirement. |
| durable-report-12 | Production retains the observed completion time plus a separate no-schedule instruction; candidate loses the history by abstaining. Both retain the future timestamp policy, but candidate omits its explicit no-fixed-schedule limitation. Separate retention of the production limitation is descriptive, not proof of ideal fact granularity. |
| mixed-marker-13 | Production retains the marker combined with the temporary tool/storage prohibition. Candidate loses the marker by abstaining. For the continuing-policy control, production retains both marker and policy; candidate retains only the policy and loses the marker. |

The candidate is not adopted. It removes several transient instructions that
production retains, but loses required historical and mixed-content facts and
some limiting conditions. Production also remains semantically inadequate.
Zero runtime errors do not close extraction quality, service recovery or T18.

The follow-up keeps all 26 cases and adds the previous 20 history/preference
cases, with their original existing-fact contexts and ambiguity review.
The preserved v2 preparation was never executed. V3 adds assertion-level
separation and observed-history guidance after these terminal failures.
No labels, thresholds or source events were changed to produce a better result.

Evidence: [terminal record](extraction-durability-development-terminal-v1/README.md)
and [v3 frozen preparation](extraction-durability-history-preparation-v3/README.md).

Reviewed input hashes:

- `proposal-review.json`: `bc9c46916f2ba2e7ea1a79248c5c68de8066d4dfd160ab1555f3a7ce60bab0c8`
- `rubric.json`: `2dc22d2ae0b851ccfaab6a9d607f72790ab0a5b028697726b95d7031333bb6a7`
- `rubric-review.json`: `263c535f8a23a8f199a970cbf1a763db65cfd94f6266bbf6338c341c3b77e2d2`
