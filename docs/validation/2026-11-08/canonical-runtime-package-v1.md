# Installed canonical runtime verification

The canonical source at `d9ab39ba6b23815c3b0457c1d389d2f70fb3fd4c` was built into a new private validation directory. The sdist was built first and the wheel built from that sdist, using isolated Hatchling 1.33.0 environments. Both retain the unpublished candidate version 0.7.0b1; earlier artifacts were not overwritten. This result includes the recent linear byte-packing change, asynchronous Codex/Claude identity lookup and raw-capture context-warning correction, alongside the existing queued-capture implementation.

Wheel RECORD hashes and sizes passed verification. All 98 wheel runtime/resource members and 105 sdist source members matched the checkout, including bundled agent files and benchmark datasets. Twine metadata checks passed. The wheel was installed into a separate Windows Python 3.12.11 environment with the prior pinned dependency set. Dependency consistency passed; this is a retained-dependency check, not a freshly resolved dependency matrix.

The test runner used Python isolated mode, asserted that `mnemos` imported from the new installed environment, and ran copied tests from a consumer directory containing no source package. It selected 37 test files covering graph persistence and transactions, scopes, forgetting, outbox/replay/service behavior, queued capture, context rendering/budgets, and Codex/Claude identity responsiveness. All 418 executed tests passed in 190.05 seconds. One additional test, `test_replaced_open_path_never_acknowledges`, skipped because Windows prevents renaming the open SQLite database. That filesystem scenario remains unverified here.

The full test-only Python tree was copied to satisfy fixture imports; 324 copied files were checked against the original sources after execution, but only the selected 37 files were executed. All 98 installed package members still matched the wheel after testing. Fixtures isolated user configuration and databases. This did not change or activate the live installation, validate interactive native-host behavior, run an MCP-extra matrix, establish cross-platform support, or satisfy the original semantic, timing and pilot gates.

Artifacts are retained under `.venv/canonical-runtime-package-v1`:

| Artifact | SHA256 |
| --- | --- |
| mnemos_memory-0.7.0b1-py3-none-any.whl | 5182601cb4f8f4739bb8472e255e2e0ee019ae27362a65829d363e7df0464a6f |
| mnemos_memory-0.7.0b1.tar.gz | fb08c58d83fe5480794ed9dc9f91ebd5238ee80a2e8a4b6ad89051047c79610c |
| installed-tests.xml | dea8efbbcc8c4a0fa41fb886495092dc360246d572d7118a3e705690d600fa91 |

The evidence archive `.venv/canonical-runtime-package-v1-evidence.zip` contains 338 members and has SHA256 `137c76dcaf3f8d5f6aec38bd049d78ebd84be9d999dd8ff5f8039a09d9abb418`. Its contents were read back and checked against the retained files. The earlier archive-verification receipt correctly records that installed tests had not yet run at that step; this report and the final receipt record their subsequent completion. Nothing was published.
