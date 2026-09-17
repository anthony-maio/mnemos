# Provisional capture package verification

Frozen source `a38ee70d19f2ce8413d1ab7c9ce2ff26ffd07b46` passed 754 tests with two optional skips, Black, strict compatible mypy, JavaScript syntax, build, and Twine. A fresh MCP-enabled wheel installation passed source/asset parity and capture replay checks. The installed Claude adapter also passed the instrumented 10-second per-command timing and restart/replay gates. The timing probe does not start Claude or a model, and excludes SessionStart.

Wheel SHA256: `94b19bb5fd758e715031a8a6d6e3599c0a6955145e5717bf9325607900096ecc`. Wheel and source archive binaries remain at the local paths recorded in verification.json; their hashes are preserved here. No release was published. These checks do not establish native host continuity or real-model quality. The earlier archived-payload formatting failure remains in [the first-attempt record](../provisional-checks-first-attempt/README.md).
