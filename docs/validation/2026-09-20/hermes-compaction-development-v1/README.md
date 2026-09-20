# Hermes compaction provenance development

The host diagnostic reproduces two distinct problems with Hermes `14a3463`
plus the existing row-ID compatibility patch and the verified Mnemos `4bbab4f`
wheel. Retained originals and concurrent watermark-tail copies become new
child rows and new Mnemos observations. Separately, an authored fixture puts
an assistant summary after a retained assistant answer; role-alternation
repair merges them and drops the summary marker. This demonstrates a loader
boundary failure, not an observed model-generated compression layout.

The original four parent messages become five child database rows. Raw
`get_messages` preserves the summary classification. The no-repair
`get_messages_as_conversation` projection omits that marker. The repaired
projection returns four rows, including the merged answer and summary, and
all four become normal child-session captures. The report field
`summary_marker_reloaded` checks the raw database representation, not the
conversation projection. Extraction was not run, so this establishes normal
adapter eligibility, not a measured extracted assertion.

The diagnostic uses fixture turns, real SessionDB publication and loaders,
actual MemoryManager, and a queued outbox. It verifies the wheel hash, source
revision, module origins, unchanged inspected host files, and zero attempted
network connections. Its first malformed report is retained. The final
runner hash matches the final report; the first attempt's runner bytes were
not retained. Neither attempt proves a true summary-plus-live-user carrier
contract; that fixture explicitly sets the live-user flag false.

The Mnemos candidate accepts a strict host-provided `_memory_origin` record:
version 1, original session ID, original positive row ID, and original cwd.
The cwd must resolve to the same registered profile/project/workspace as the
active session. Copies reuse the original context, native event ID, and
occurrence timestamp. Malformed provenance and cross-project redirection fail.
This requires a host compatibility change; the current host does not yet
provide this record.

Marked summaries remain provisional, capture-only evidence. Durable summary
carriers use `provisional:v2:summary:<row_id>` source references and share the
ordinary native carrier's event ID. This prevents dropping or adding the
summary marker from silently minting a second occurrence. A later conflicting
classification or content is rejected by the outbox/service. Missing native
identity retains the existing provisional-receipt limits. No prior captures
or identities are migrated.

The initial 10 adapter regressions fail on the baseline. Adding direct
service checks produces 12 failures. The first repair passes 38 focused cases.
A further pair exposes classification changes minting new identities; the
shared-carrier correction passes 40. The expanded run initially has one test
fixture error because it omitted `forget_fact`'s required version argument;
that result remains recorded. Supplying the actual version produces 76
passing cases, including 21 new cases. They cover restart, replay, source
conflicts, scope, summary eligibility, malformed identities, and forgetting
before a child replays the original. Black and strict mypy also pass for the
changed modules; their tool output is not archived as a raw log here.

The index binds 27 artifacts, including original XML results, failing test
inputs, the candidate source, and host diagnostic inputs/results. Run
`python verify.py` to check their hashes and the recorded outcomes.

This is focused development validation. Full package verification and the
host compatibility candidate remain separate. Role-repaired model text must
not be used as a substitute for original memory evidence. Compaction
provenance, true hybrid carriers, native conversations, and live-host adoption
are not closed by these tests.
