# Installed Git discovery stdin fix

Status: isolated installed validation passed. No latency improvement, canonical promotion, publication or live activation is claimed.

Built a new wheel and sdist from capture candidate 98450b397bbaaca0d09c756eb91e40818d3145a8 with hatchling 1.33.0. The wheel contains the stdin isolation fix alongside the experimental SQLite worker backend. All 99 package source/resource members matched source and installed bytes; all 106 sdist source members matched source. The fresh Python 3.12.11 installation retained the previous dependency versions.

The installed suite passed 165 tests in 115.41 seconds. One test skipped because Windows prevents renaming an open SQLite database. Coverage includes the process-level input-isolation regression, project identity resolution/cache and scope checks, queued capture receipts, JSON/SQLite replay, real-service correction and forgetting without resurrection, lost acknowledgments, and service failure handling. Nineteen copied Python test files, including the package marker, matched source hashes. Post-test installed parity passed.

The first installed run passed 163 tests but failed two SQLite subprocess checks with ModuleNotFoundError. The runner loaded the installed package, but its working directory was the mnemos improvements repository; child Python processes inherited that directory and found its older source package. The revised runner changes to the isolated artifact directory before testing. A separate child-process import check confirmed the installed SQLite module path. Both runners and both JUnit files are retained. No test or product behavior was changed to suppress those failures.

Wheel SHA256: bbfc09d149959669c0098c01f82dcda409f3566cfebd603dff740a6c16fe29be.

Sdist SHA256: 5d9304c16146722411814c5d261fffd3374be01123c0c5030f2070239de9d05a.

The hash-verified 36-member evidence archive contains artifacts, constraints, dependency manifest, test copies, both runs, import evidence and verification helpers. Archive SHA256: c97674cfbf36484025a4933f02d95f09b793c6fc8fe2e01d56d37df019b4eb74. Exact counts and hashes are in the adjacent JSON receipt.

The previous v4 artifacts and all installed files were independently rechecked unchanged. Their timing results remain attached to v4. V5 still needs a separate bounded check of the original stdin-controlled owner scenario before claiming that particular stall is fixed. The original 100 ms command target, retrieval and quality gates, fresh acceptance and actual daily-use requirements remain open. The optional live pilot remains unchanged.
