# Concurrent support execution checks

The isolated candidate at `2a065f04f78e024dd6035e46fa0a2add54e9b83e` adds opt-in concurrent execution to the two-selector support composition. It descends from canonical delivery commit `c77933402a8bb4629211f24be22c3555fc872a76`. Sequential execution remains the default, and the canonical delivery worktree is unchanged.

All 78 tests in the answer-support, intersection, and new concurrency suites passed. The 14 new cases cover overlap without depending on timing speedups, stable component order, exact decision/evidence parity, and cleanup after provider failure, configuration failure, stale snapshots, timeout, and caller cancellation. Both selector positions are tested for failure cleanup. Strict typing passed for the changed module; formatting passed for both changed Python files.

Concurrent calls receive the same immutable snapshot. Both still have to select a fact for it to be returned. The implementation preserves selector order when assembling results and cancels and awaits unfinished sibling tasks on every exit. It retains scope and snapshot checks around provider work. The shared timeout depends on cooperative cancellation, as it does for the sequential implementation.

No model inference, semantic evaluation, production configuration change, or pilot activation was performed. These tests establish scheduling and lifecycle behavior, not a latency improvement or better answer support. A shared inference server may serialize concurrent requests or become slower under contention. The previous quality and latency failures remain open.

The archived live-composition helper uses an awaited QA worker request, so its adapter has an asynchronous boundary where overlap may be possible. That source inspection does not establish cancellation behavior or provider capacity. The next check must verify the worker transport's cancellation contract before a prospective latency experiment. Closed evaluation fixtures must not be rerun to replace their original results.
