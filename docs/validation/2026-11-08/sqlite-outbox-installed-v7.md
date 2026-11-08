# Installed SQLite replay batches

The fresh v7 capture-candidate installation passed 42 selected checks in 37.51 seconds, with one expected Windows skip because the operating system prevents renaming an open SQLite database. There were no failures or errors. Coverage includes batch commit rollback, cancellation with original event IDs preserved, concurrent appends, compatibility replay, lost acknowledgments, and real-service correction and forgetting.

Built from 7af4de9df8a33a21d984dad827878e7950b36180 using hatchling 1.33.0. All 99 wheel source/resource members and 106 sdist members matched source. The fresh Python 3.12.11 environment used the prior dependency versions. All 99 installed members and ten copied test/fixture files passed hash checks after execution. The runner used the artifact directory as its working directory so child processes imported the installed package.

Only mnemos/client.py and mnemos/sqlite_outbox.py changed among the 99 package files compared with v6. The previous wheel and sdist remain unchanged. This is focused installed regression validation, not a new full-suite result.

Wheel SHA256: 08421819844a85c46e4f570266a2fb26030e3701e6249be6f141cb53b091151b.

Sdist SHA256: 7d6b4c5e34839970bbd7ec8b2bc2c5d160c5e7e2e1ecf623a4aaa15261549215.

The verified 25-member evidence archive retains the distributions, copied tests, JUnit, parity records and audit helpers. Archive SHA256: f3bd7786de44a1713a5db18f8e4c08321fedfafe22f923c1cf41653b9fa4e0e6. The adjacent JSON contains the machine-readable receipt. The test session handle was no longer available on recovery; the completed JUnit establishes the test outcome, rather than a recovered shell exit status.

No performance improvement is claimed. A prospective backlog replay comparison remains necessary, and replay throughput does not establish the 100 ms durable-capture requirement. The candidate remains isolated from the canonical runtime and live pilot. The original quality, latency, fresh-acceptance and daily-use gates remain open.
