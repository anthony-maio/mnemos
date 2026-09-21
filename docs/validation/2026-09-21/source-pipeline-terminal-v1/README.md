# Raw-source pipeline development v1

The first combined selection/reconciliation run uses sixteen frozen authored development cases with local Qwen2.5:14b at both stages. The preparation binds the model digest, candidate 0560027, historical provider source, exact inputs and configuration. No supplied selection labels enter the pipeline; no facts or review records are written to a database.

Fifteen cases are runtime-valid; foreign scope rejects before either provider. There are fifteen selection calls and twelve reconciliation calls. Six exact-source FactProposal values and seven explicit review outcomes are returned. Thirteen of sixteen frozen operation/content expectations match. This includes intentional unresolved results, discarded inputs and a scope rejection; it is not semantic accuracy or a release gate.

Three failures remain: the staging-host fact with an unrelated target is unresolved; the Tuesday-window candidate in a mixed support/add source is unresolved; and the complete negative side-panel rule is classified as correction_or_rejection and remains unresolved instead of being retained. The stronger prospective-rule expectation was frozen before this run. It is not replaced with the earlier supplied-kind unresolved expectation.

The two pure prospective negative rules are retained, including separation from a one-turn formatting instruction. A question, an assistant suggestion and a temporary heading instruction are discarded and never reach reconciliation. Referential rejection with no replacement remains available as an explicit review outcome.

Additional semantic limitations: the multi-candidate source is labeled user_preference/continuing_requirement rather than the original asserted facts. Its exact wording survives, but those labels are not established by the source alone. The assistant conflict reason calls its content direct user content even though source_role=assistant and source_kind=attributed_history remain intact. The reason is diagnostic text and never becomes FactProposal content, evidence or authority.

The orchestration passes 110 combined focused tests and Python 3.10 type checking for its module/tests. It is experimental and write-free. Durable review handling, real database integration, held-out quality, all-module recovery, native host acceptance and the pilot remain open.
