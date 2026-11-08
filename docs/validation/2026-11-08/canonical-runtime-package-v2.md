# Installed authorization-fix package verification

Canonical source `25126632fa9b981faf010541833abea2bc3246a1` was built into a separate private 0.7.0b1 package. It includes file-owned registry refresh, authorization revalidation after asynchronous fact staging, and revalidation before returning retrieval results. The previous package and its measurements remain unchanged.

The sdist was built first, followed by the wheel from that sdist, in isolated Hatchling 1.33.0 build environments. All 98 wheel runtime/resource members and 105 sdist source members matched the checkout. Wheel RECORD hashes/sizes and Twine metadata checks passed. A separate Windows Python 3.12.11 environment installed the wheel with the retained pinned dependency set; all 43 distributions matched that set, and dependency consistency passed.

Python isolated mode loaded the installed package and ran copied tests from a consumer directory without a source package. The 42 selected files cover authorization regressions, retrieval views, graph persistence, forgetting, service lifecycle, outbox/replay, host adapters and context behavior. Results: 471 passed, one skipped, in 148.79 seconds. The skip is the existing open-SQLite-path replacement test: Windows prevents the filesystem operation. The previous Linux result exercised that scenario on the older wheel, not this artifact.

All 325 copied test Python files were compared with their source files after execution; only the selected 42 files were executed. All 98 installed package members still matched the wheel. The evidence archive was read back and its 339 members verified against retained files.

| Artifact | SHA256 |
| --- | --- |
| Wheel | 9f7a93f616b4af680bee576fbc70ddebf8dd9dc7aeae2f4f5f8fb085c7079b69 |
| Sdist | 512a1e399b61d5bf8dc8b62c0246034292e357277c31ef1e6d91aeb5e2dc8aca |
| JUnit | 7224ceec22b66266808c9f1ec7ed0837e1c4aaeff3d3741cb27503933710f5e4 |
| Evidence archive | 39b94807ad0b228addd9173e74bd867b09cded8ba998162f8608855c11e1bf17 |

Private artifacts are under `.venv/canonical-runtime-package-v2`; the archive is `.venv/canonical-runtime-package-v2-evidence.zip`. The [machine-readable receipt](canonical-runtime-package-v2.json) records the build and installed results separately. This is not publication, live activation, a native-host acceptance run, a new latency comparison, a quality pass or actual pilot use. No live registry, memory database or optional-pilot installation changed.
