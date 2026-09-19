# Capture storage placement diagnostic

The unchanged capture fixture on source `39b9c79` passed its 100 ms p95 gate
in all three C: SSD repetitions and failed in all three D: HDD repetitions.
All 300 measured events committed. Each repetition used 50 captures after
one warmup, simple embeddings, unavailable cognition, WAL, and FULL SQLite
synchronization. This is capture-only synthetic-provider evidence.

| Run | Volume | Capture p95 ms | 100 ms gate |
| --- | --- | ---: | --- |
| 1 | C: | 31.853 | PASS |
| 2 | D: | 141.634 | FAIL |
| 3 | D: | 521.719 | FAIL |
| 4 | C: | 31.795 | PASS |
| 5 | C: | 35.024 | PASS |
| 6 | D: | 398.411 | FAIL |

The prospective order was C, D, D, C, C, D. Recorded hardware maps C: to an
ORICO NVMe SSD and D: to a ST4000DM004 SATA HDD. This shared-machine diagnostic
ran alongside real-model evaluation and development I/O, including D: Git
activity during the later repetitions. It is not an idle-device comparison;
the full tail variation cannot be attributed solely to physical media.

Separate instrumented runs measured capture p95 at 33.081 ms on C: and
131.099 ms on D:. On D:, fsync p95 was 70.928 ms and SQLite capture p95 was
50.994 ms, versus 4.536 ms and 2.826 ms on C:. Nested stage durations overlap
and must not be added. These profiles identify durable I/O as a substantial
component of this observed difference; they do not justify weakening durability.

The original local performance failure is preserved: capture p95 135.186 ms
and recall p95 4,063.309 ms. Its report did not record the actual temporary
storage path. D: placement for that original run is inferred from the launch
environment, not established by its report. These new runs have explicit paths.
Recall performance, native-host capture, and daily-use acceptance remain open.

The subsequent benchmark change at `7bcabde` adds `--work-dir`, records actual
storage paths and filesystem device IDs, retains explicit fixtures, and refuses
existing work directories or report paths. It changes no timing threshold or
durability setting. Filesystem device IDs do not themselves identify hardware.

The original commands were `.venv/Scripts/python.exe` followed by each of
`.venv/profile_capture_stages_v1.py`, `.venv/profile_capture_stages_v2.py`, and
`.venv/run_capture_placement_v1.py`, from the implementation worktree. The
archived helpers retain their original absolute paths and reject reused output
directories. Source binding and helper hashes are checked before copying.
Raw stage events, every per-capture latency, hardware observations, and all
failed repetitions remain in the JSON reports. No live data or configuration
was changed. Existing diagnostic SQLite fixtures remain at their original paths.
