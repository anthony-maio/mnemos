# Raw-source pipeline development v2

Candidate e449bcc combines the relationship-label adapter with stricter correction/conflict ambiguity guards. The sixteen raw-source cases, their frozen expectations, classifier, Qwen2.5:14b settings and provider snapshot are unchanged from pipeline v1. Every classification inventory is identical to v1. No source labels are supplied to the pipeline, and no database writes occur.

Fifteen cases pass runtime validation; foreign scope rejects before either provider. Fifteen selection calls and twelve reconciliation calls produce eight exact-source proposals and five explicit review outcomes. Fifteen of sixteen frozen operation/content expectations match, including intentional unresolved/discarded outcomes and scope rejection. These are development comparisons, not a semantic accuracy score or release acceptance.

Both independent-fact omissions are repaired. Staging-host information is retained alongside an unrelated production-host target, and the mixed source produces support plus add. The ambiguous correction stays unresolved. Assistant conflict remains attributed and its explanation no longer claims direct user content.

The remaining failure is the complete negative rule introduced by rejecting an old rule. Selection still calls it correction_or_rejection and reconciliation leaves it unresolved; the stronger retention expectation remains unchanged. Pure prospective negative rules and temporary/question/suggestion exclusions still work. The imprecise user_preference/continuing_requirement labels on the mixed factual source persist, even though exact text is preserved.

Thirty-nine focused reconciliation/relationship/pipeline tests and Python 3.10 reconciliation-module type checking passed before the run. Storage integration, durable review lifecycle, fresh semantic acceptance, all-module recovery, host acceptance and the pilot remain open.
