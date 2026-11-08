# Capture candidate full source validation

Candidate 0684c9839ff3c1a0cc76fcaa665cf9973ce0eb14 passed 3,761 tests with 81 skips and no failures or errors. Pytest exited zero after 932.41 seconds; its JUnit suite records 923.101 seconds. The complete source suite includes the queue candidate and the repository's existing development checks. It does not establish fresh semantic acceptance, latency requirements or the daily-use pilot.

The first attempt stopped with 119 collection errors because the sparse checkout omitted benchmark modules and the focused installed-package environment lacked jsonschema and MCP. That failed JUnit remains preserved. The second attempt populated the tracked checkout and used a separate Python 3.12.11 environment with the v7 dependency versions plus jsonschema and MCP. Both the base requirements and resolved environment are saved. The validated v7 environment was unchanged.

The 81 skips comprise 75 unconfigured experimental native-launcher cases, one unconfigured website check, two WSL bridge cases, two POSIX shell cases and one Windows open-SQLite rename case. A separate follow-up configured the previously validated native binary and canonical website path, then passed all 87 tests in those native and documentation modules in 15.73 seconds. This exercises 76 previously skipped cases plus 11 repeated documentation checks. The five platform-dependent cases remain unrun here.

The native binary hash was checked before the follow-up: ec52c79a0ae613ae336836b3e82ddf853a601a0de3f7cd3aa0047318d06ee12b. Follow-up JUnit and log are retained at `.venv/capture-native-followup-v1.xml` and `.venv/capture-native-followup-v1.log`.

Follow-up JUnit SHA256: 9a2547effc9fa3cea55879a907445709403fcd009c09e81c28da161f82cbaefb.

The full-suite archive retains both JUnit attempts, the completed log, dependency records, prepared patch and audit helper. Its seven members passed hash and ZIP integrity checks; SHA256 6fa35925c3cfbd2e8c0728c6988ff00ede9591c12770eaf856a69483e08d3cb0. The adjacent JSON lists every skip and input hash. The follow-up is separate from that frozen archive.

This evidence supports proceeding to local delivery integration. Validation of the actual delivery tree and its newly built package remains required. Live host configuration, the optional explicit-memory pilot and publication state remain unchanged.
