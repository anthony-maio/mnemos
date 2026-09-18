# Competing service recovery

A dead owner can leave its descriptor behind. While one replacement holds the process lock and initializes its dispatcher, a second launcher previously accepted that old descriptor and exited successfully. The waiting client then failed before the actual owner was ready. The original regression failed because the second launcher returned during the controlled initialization barrier.

A competing launcher now uses authenticated GET /v1/health before adopting an owner. It polls absent or unreachable descriptors within the startup wait, with a per-operation HTTP timeout capped at 250 ms and a 4 KiB health-response limit. Authentication rejection or invalid health responses fail explicitly. It never dispatches memory operations, rewrites the descriptor, or initializes a second dispatcher.

Five added cases cover delayed replacement, dead-owner timeout, rejected authentication, a boolean schema version, and an unready status. The timeout and refusal cases verify that the descriptor stays unchanged. The complete source suite passes 1,054 tests with four skips. XML reports and before/after source are indexed here. The initial 22-case intermediate check is not used as final evidence because the regression file was expanded while that process was running; the focused and full runs began afterward.

This archive does not close simultaneous native-host recovery or configured-provider acceptance. It uses real process locks and authenticated loopback HTTP with a fixture dispatcher. The installed package checkpoint remains 37e446f until the changed source is rebuilt and checked separately. Real-model evaluation remains on its frozen source.
