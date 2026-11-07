# Temporal support failure review

The completed [v4 comparison](answer-support-temporal-v4.md) does not support
promoting the temporal treatment. This review reads the saved results and
source facts; it makes no new model requests and changes no scores. The
[audit receipt](answer-support-temporal-v4-failure-review.json) binds the private
failure ledger and lists every fully assessed vector-arm error. That arm had
no provider failures, so these examples isolate selection errors from timeouts.

| Failure | Saved behavior | Implication for development |
| --- | --- | --- |
| Wrong attribute | Joint support returns a frontend region alongside the requested backup region; qualified support removes the extra fact | Subject overlap alone is insufficient |
| Missing requested value | Qualified support returns a queue technology for a cloud-region query, a verification requirement for a record identifier, and a migration rule for an approver | The requested value must actually be present |
| Missing historical evidence | Both treatments answer a 2020 handbook-location question with an undated current fact | Current wording cannot establish a historical state |
| Missing future evidence | Both use a November 2026 backend status for January 2027; joint support also selects quoted fixture instructions | A dated state and quoted instructions do not establish a future change |
| Unresolved contradiction | Both select 30-day and 90-day retention claims for the same current policy | Individually sourced facts can still conflict as an answer set |

Every wrong selection in this slice reports probability 1.0. Those model
probabilities do not establish calibration; increasing the 0.8 threshold would
not remove these errors. Retain the frozen threshold and results.

The qualified treatment fixes one attribute error and rejects the quoted future
instruction, but it adds three missing-value failures. The longer instructions
may distract from the requested relationship, but this audit cannot establish
that cause. Source presence and runtime-valid output do not establish that a
fact answers the question.

The next development treatment should distinguish three decisions: whether
the candidate supplies the requested value, whether its source supports that
value with the requested time and qualifications, and whether visible evidence
leaves the answer unresolved. Each decision needs explicit negative and unknown
outcomes. Keep the existing scope, source binding, revision fences and request
limits. Do not add entity-specific rules from these examples.

Freeze any new treatment before comparing it with the joint baseline on the
same inputs. Report supported hits, clean hits, false support, abstention,
provider failures and latency separately. These inspected questions remain
development data. Fresh evaluation, extraction quality and pilot entry gates
remain open; this review satisfies none of them.
