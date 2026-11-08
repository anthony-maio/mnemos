# Functional plugin Python selection

The separate source repair at `d854db7e` fixes command overrides in both
functional plugin entry points. `MNEMOS_PLUGIN_PYTHON=python` now resolves
against the configured PATH before subprocess launch. This prevents the
Windows base interpreter from replacing the selected virtual environment.
Explicit paths, Windows executable extensions and Linux virtual-environment
symlinks retain their intended behavior.

An unavailable command produces an MCP diagnostic. The hook reports
`runtime_not_ready`, skips without installation and returns a nonblocking
status. That skip does not count as successful capture.

## Verification

The initial 12 regression cases produced eight failures and four passes.
After the repair, the focused suite passed all 38 checks. Exact committed
copies of those three test files then passed in six environments:

| Environment | Python | Passed | Failed or skipped |
| --- | --- | ---: | ---: |
| Windows x64 | 3.10.18 | 38 | 0 |
| Windows x64 | 3.11.13 | 38 | 0 |
| Windows x64 | 3.12.11 | 38 | 0 |
| Debian 13 under WSL x64 | 3.10.21 | 38 | 0 |
| Debian 13 under WSL x64 | 3.11.15 | 38 | 0 |
| Debian 13 under WSL x64 | 3.12.13 | 38 | 0 |

These are 38 unique wrapper cases, executed 228 times across the matrix. The
older environments retain their legacy package; the standalone tests omit
the functional package's conftest and establish wrapper behavior only.

The unchanged functional wheel was installed into a fresh target on Windows
Python 3.12.11 using the existing integration dependencies. Five checks passed:
four real MCP roundtrips through generic/desktop configurations and the plugin
with absolute-path and command overrides, plus one Codex config/guide check.
The roundtrips exercise store, retrieve, evidence inspection, feedback,
correction, forgetting, health and resources with temporary scoped SQLite
services, SimpleEmbeddingProvider and MockLLMProvider. External sockets were
disabled. This is subprocess transport coverage, not native application
acceptance or a new semantic-quality result.

All 96 installed package files match the frozen wheel before and after the
run. All 34 imported memory modules resolve inside the isolated target, and
dependency versions are unchanged. Strict typing passes for both wrappers;
formatting passes for all four changed Python files. The 14 watched live and
inactive-pilot paths are unchanged.

## Delivery state

The repair changes five files relative to canonical source `3c833caf` and
remains on its own branch. Its binary patch reconstructs the exact committed
tree. An offline PR description and private evidence archive accompany the
[machine-readable receipt](functional-plugin-python-v1-receipt.json).
The memory package, manifests, hook configuration and canonical checkouts are
unchanged. No wheel was rebuilt or published.

The original quality and timing failures still apply. Full native hook
validation and both pilot scopes remain pending, with zero observed pilot
days. This check does not establish full package support on all six platforms,
native Linux CI, macOS support or release readiness.
