# Capture interruption package checkpoint

Frozen source `b7b4d6bdca19994ee4b3336d3dd166d30da6910c` passed 958 tests with four platform skips, Black, strict mypy, JavaScript syntax, build, and Twine.

A fresh installed wheel passed 264 selected cases with two matching platform skips and seven replay assertions. Exact case identities, outcomes, and skip reasons match the frozen source run. The installed runner sets PYTHONSAFEPATH=1 so Python subprocesses do not import the source checkout merely because it is their working directory.

Seven new cases verify sanitized capture persisted before network waits, forced client termination before send and after commit, restart replay without duplicate logical events, terminal rejection cleanup, ambiguous reply retention, and forget suppression while capture remains buffered. The repaired client does not establish native hook deadline or host shutdown acceptance. Real-model quality, latency, and pilot gates remain open.

The first installed run exposed a test-origin gap: the interruption test could import the source checkout in its child process. The test now binds both processes to the same client module path. Its seven cases pass again in source and against this exact installed wheel. The supplementary report verifies the loaded client bytes against the wheel and unchanged product source. The original reports remain preserved; the supplementary origin-bound run supplies the child-process installed evidence. No product code changed for this validation repair.
