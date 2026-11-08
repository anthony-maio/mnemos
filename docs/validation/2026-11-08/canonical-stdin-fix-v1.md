# Canonical Git discovery input isolation

Canonical delivery source at 5c8aa79c includes the two-file stdin isolation fix from capture candidate 98450b39. Git project discovery receives DEVNULL stdin so a discovery subprocess cannot consume the host's protocol/control stream. Git arguments, environment filtering, timeout and fail-closed identity behavior remain unchanged.

All 27 project identity resolution and scope context tests passed against canonical source in 6.35 seconds. This includes the process-level input-consumption regression. Tests used a fresh temporary directory outside Git repositories. The changed scopes module exactly matches the tested capture candidate; the adjacent receipt binds the JUnit hash.

The isolated capture candidate additionally passed 165 installed checks and six captures using the original stdin-controlled owner setup. Those results belong to its v5 wheel, which also contains experimental queue changes. They are not canonical package validation. Canonical artifacts must be rebuilt and checked before delivery.

The copied backend comparison retains both failed 100 ms results: JSON command p95 690.03 ms and SQLite 476.37 ms. JSON remains canonical; SQLite and queued worker changes have not been integrated here. All historical failures and original gates remain intact.

This is local preparation only. No public branch, release, hosted page, live pilot configuration or memory database was changed. Quality, performance, fresh acceptance and actual daily-use obligations remain open.
