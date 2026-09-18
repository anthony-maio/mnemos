# UUID write safety validation

Terminal result: passed. Full-suite totals: {"tests": 1013, "failures": 0, "errors": 0, "skipped": 4, "passed": 1009}.

The shared firewall recognizes complete canonical UUID identifiers when classifying numeric phone, SSN, and card matches. It ignores only matches wholly inside those UUIDs. Personal data outside or crossing their boundaries, email, and credential assignments still use the configured policy.

The regressions cover UUID variants and boundaries, block/redact/allow policies, adjacent sensitive data, client outbox identity, and durable remember/correct/replay. The initial focused run before implementation had 13 failures and nine passes; a later expanded safety/module/capture run passed 116 tests. Those tool-observed counts are distinct from the source-bound full-suite logs archived here.

The earlier full-suite empty-fact failure remains preserved. Its generated ID was not retained, so the deterministic UUID reproduction does not prove that failure's particular cause. Native host acceptance and real-model evaluation were not rerun by this check.
