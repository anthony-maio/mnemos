# Condition-query delivery alignment

Canonical delivery commit `06e3cf70` adds the closed condition-query report and receipt and describes the failed result in the README. Website draft commit `594c2b1` adds the matching receipt and evaluation-page explanation. Both distinguish structural validity, predicate correctness, final applicability and timing, including the two false establishments hidden by conjunction.

The two copied canonical files and website receipt matched the source bytes. The website receipt SHA256 is `ba32a7638ddfd0d62104e7acdc56e79962c5d2e59a39848bc4aa6be967ac28bf`. The evaluation page has exactly one link to that receipt. Source metrics were checked before writing, including 26/30 predicate states, 15/16 applicability results, 9/11 adequate witnesses, 3/4 applicable cases retained and 1781.68 ms provider p95.

All 12 canonical documentation-consistency tests passed in 0.29 seconds using the local website path. JUnit is `.venv/condition-docs-v1.xml`; the byte/link/metric audit is `.venv/condition-alignment-v1.json`. Diff checks passed and both worktrees were clean after their scoped commits. These checks establish local documentation alignment, not hosted rendering or publication. No push, deployment, live configuration or package change occurred. Original quality, complete timing, fresh acceptance and pilot gates remain open.
