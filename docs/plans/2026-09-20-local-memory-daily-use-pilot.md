# Local-memory daily-use pilot

Status: NOT STARTED. This protocol prepares the two-week, two-project pilot
required by the functional-local-memory plan. It is not evidence of daily use.
The current extraction and provider-recovery failures remain entry blockers.
A local personal pilot does not satisfy the separate external pilot requirement
in the release checklist.

Before starting, bind one verified source commit, installed wheel hash, model
digests, provider options, host versions, project identities and module settings
in the [pilot record template](../pilot/local-memory-pilot-record-template.json).
The same checkpoint must pass source/package and installed checks, scoped
lifecycle/recovery checks, and the applicable quality gates. Keep neutral or
negative module results visible; this protocol does not enable experimental
modules by default or replace the real-model ablation requirements.

Live host configuration and live memory migration require Anthony's explicit
authorization under the active task. Prepare the exact configuration diff,
backup/restore result and rollback instructions for review before requesting
that authorization. Do not start the pilot clock or modify live configuration
merely because this protocol exists. Publication is a separate action.

1. Select two real projects and the installed hosts that will participate.
   Record their actual project IDs, host versions and supported capture paths.
   Mark unavailable host paths as untested. Keep the project names private
   in the research export if their work is not public.
2. Start the dated record when actual daily use begins. Keep at least two
   weeks of daily observations across those projects. Record missed days and
   outstanding coverage honestly; do not fill them with synthetic sessions or
   count earlier automated test runs as pilot days.
3. During normal work, note useful memories, missed memories, wrong memories
   and interruptions. For planned recall checks, write the expected fact and
   its source before asking the question. Keep exact source/event and fact IDs
   when available so corrections and repeated arrivals can be distinguished.
4. Exercise the lifecycle with clearly disposable project facts: teach in one
   host, recall after a service restart in another, correct the same fact,
   inspect its version, forget it deliberately, and confirm absence after
   restart and exact original-source replay. Include an unrelated retained
   control and a foreign-project query. Label these checks as scripted pilot
   checks, separately from natural daily-use observations.
5. For an actual interruption, retain the capture receipt/outbox state, job
   attempts, provider error and recovery outcome. Simulated interruptions must
   use disposable data and be labeled simulated. A completed job that creates
   the wrong fact is a failure, even if transport and retry behavior succeed.
6. At the end, review every recorded miss and incident against its original
   source. Report sample counts, unresolved cases and missing evidence. Report
   observed latency separately from the 10,000-fact benchmark; ordinary daily
   use does not establish that benchmark's percentile or scale requirements.

Use one daily entry per observed day. Its fields should identify the date,
projects used, hosts used, actual work, recall checks, lifecycle checks,
incidents and evidence locations. An empty list means no recorded evidence,
not that a behavior passed. A useful recall record includes the expected
meaning, source reference, returned fact IDs, whether the answer preserved
scope/time/uncertainty, and the reviewer's judgment. Runtime-valid output alone
is insufficient.

The completion review must establish the original two-week/two-project scope
and distinguish natural use from scripted checks. Unresolved scope leakage,
corruption, lost acknowledged capture, forgotten-fact resurrection, or startup
failures leave readiness open. Preserve failures and their repair checkpoints.
If a repair changes behavior during the pilot, record the boundary and give the
repaired behavior additional observed coverage; do not pool versions silently.
The original quality thresholds remain those in the implementation plan.

Keep the unredacted evidence local. The Substack handoff receives a reviewed
summary with publishable examples and exact evidence limits, not private
conversation transcripts by default. Completing this protocol does not itself
authorize a release or publication.

Sources: [implementation plan](2026-09-16-functional-local-memory.md),
[release checklist](../release-checklist.md), and
[host compatibility matrix](../client-compatibility-matrix.md).
