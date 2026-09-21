# Support proposals must refer to the observed fact revision

A support proposal previously named a fact without requiring the version shown
to the model. The service then reread its current content. A concurrent correction
could therefore receive evidence intended for the old revision. Conflicting
support and correction operations also committed in the same extraction batch.

Sleep extraction now copies the supplied fact snapshots and binds support to
their observed versions. An omitted provider version is filled from the trusted
snapshot; an explicitly wrong version is rejected. The service checks that
version before using the target and supplies it to the existing atomic SQLite
commit check, covering changes during embedding. Support and correction cannot
mutate the same target in one extraction batch.

The four extraction-version tests reproduce three failures and one passing
control on immutable b984411. They pass with the repair. An initial combined
selection passed 22 tests, followed by 52 service, correction, forgetting and
maintenance tests. Formatting and Python 3.10 type checks pass for the changed
production files. The expanded SQLite file reproduces three failures and two
passing controls against immutable b984411: stale support attaches after the
model snapshot, and mixed support/correction commits in either order. The
embedding-window safety test already passes on that baseline, so it is not
claimed as a reproduced resurrection defect. The no-race control also passes.
See the [baseline transcript](support-revision-baseline-v1.log).

All nine new extraction and service cases pass with the repair. The successful
support control checks unchanged content/version, attached source evidence and
job completion. The [full source checkpoint](support-revision-source-checkpoint-v1/README.md)
passes 2,144 tests with four skips and all six source/package checks. The fresh
[installed check](support-revision-installed-checkpoint-v1/README.md) passes
868 selected cases with four matching skips, origin checks and the replay probe.
The earlier 2,135-test checkpoint predates this change.

This is a concurrency repair, not evidence that model-selected support is
semantically correct. No live memory migration or host configuration changed.
