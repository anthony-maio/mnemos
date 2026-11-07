# Duplicate fact identity repair

A direct user candidate that exactly matches multiple fact identities now
requires an unresolved review. The decoder cannot offer a disagreement or a
different correction target to avoid that ambiguity. Runtime validation rejects
those alternatives even when a provider ignores the schema. The judged path
returns the same review without asking the model to resolve an identity the
source does not distinguish.

The repair is committed at
`c30ed298188dd3402417c02e320794d2c696f8ad`. It follows a mechanical gap found
in the [reconciliation comparison](reconciliation-v1.md), where both direct
instruction variants called duplicate support a disagreement. The documented
contract already required an unresolved result for multiple matching identities.
The implementation now enforces that requirement across user source kinds.
Non-user attributed history and candidates that do not match the duplicated
content retain their existing rules.

Fourteen regressions failed before the schema/runtime repair. Six further
checks exposed unnecessary judged calls, including a correction that could
select an unrelated target and then fail the new runtime guard. The judged
path now resolves known exact-match ambiguity before making those calls.
All 142 focused checks pass, including 24 new tests. Coverage includes source
reconciliation, relationship variants, judged decisions, pipeline rendering,
service extraction, persisted review resolution and requirement revisions.
Black and strict mypy pass for both changed source modules.

A read-only replay checked all 240 original model choices against the repaired
schema and runtime. Both reject the six duplicate-support disagreements;
the other 234 outcomes remain unchanged. No model was called. Rejected stored
outputs are not newly correct model decisions, so the original comparison
scores and rejected-variant decision remain unchanged. Ambiguous qualifiers
and wrong semantic targets still need separate work.

The [receipt](duplicate-identity-v1-receipt.json) binds the source, tests,
pre-repair failures, final results, exact patch and replay in 13 verified private
archive members under SHA-256
`828a858df95ba689d3af940c918b287e24f066f950eb6fe38841e565baf7e4e5`.
The original comparison archive also passed hash verification. This is a local
contract repair; the installed pilot artifact, live configuration and published
package have not changed. Fresh model quality and actual pilot use remain open.
