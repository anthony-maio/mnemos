# Claude identity discovery and worker responsiveness

Claude's direct recall, session/prompt handler and transcript-capture handler now perform project identity lookup through `asyncio.to_thread`. Each lookup still calls the existing resolver. The second identity check before recall after capture remains in place. No cache or identity-policy shortcut was added.

Six regression cases reproduced event-loop blocking on the previous source. They exercise direct recall, SessionStart and the transcript-capture branch, each with normal completion and cancellation. A gated lookup prevents progress until released; the old implementation prevented the observer task from running until discovery returned. All six failed before the change and passed afterward without changing their assertions.

The tests establish that the event loop progresses while lookup is pending, no client retrieval or transcript capture begins before identity resolution, and cancellation discards the late result. Normal completion reaches the intended operation. Discovery receives a copied payload. Existing capture and recall semantics remain covered by the selected adapter suites.

All 120 tests passed in 45.55 seconds across Claude responsiveness, adapter, identity, capture resilience, provisional capture, resident hook worker and adapter context authority. Black and diff checks passed. Frozen JUnit records are `.venv/claude-responsive-red-v1.xml` and `.venv/claude-responsive-green-v1.xml`; the latter run completed with exit code zero.

This changes responsiveness, not Git execution speed. Cancellation does not terminate the thread or child process; their existing discovery timeouts remain in effect. It does not prove complete-command latency, interactive native-host acceptance or daily-use reliability. The source change is local and unpublished. Frozen packages, live hosts, memory databases and the optional explicit-memory pilot remain unchanged.
