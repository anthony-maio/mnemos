# Functional plugin interpreter repair

Carry the reproduced Windows interpreter-selection repair into the functional
plugin on a separate branch from canonical code `3c833cafe019f5ead70e83db8d70b8af4eff4111`.
Both the MCP wrapper and hook wrapper must resolve command overrides from PATH,
preserve explicit paths and virtual-environment symlinks, and exclude an
unlisted current directory. An unavailable command must produce an MCP error
or an explicit nonblocking hook skip. A hook must never install its runtime.

The initial 12 regressions produced eight expected failures and four passes.
After the repair, all 38 focused plugin checks pass. Retain both results.
Freeze the source repair before further verification. Do not change the
functional package, manifests, hooks configuration or frozen wheel.

Run the exact three focused test files from a small committed-source export
in the six existing environments: Windows Python 3.10.18, 3.11.13 and 3.12.11;
Debian 13 under WSL with Python 3.10.21, 3.11.15 and 3.12.13. Omit the main
conftest because these are standalone wrapper tests and the older environments
contain the legacy package. These cells establish wrapper behavior only.
Record every result and source hash. Do not repeat a passing cell.

Separately install the existing functional wheel into a fresh isolated target,
without dependency resolution or changes to the existing test environment.
Use Windows Python 3.12.11 and the recorded integration dependencies. Put that
target first on the isolated runner's import path, verify all 96 package files,
and verify imported package origins before and after the MCP tests. Exercise
both absolute-path and command overrides through the actual plugin manifest,
plus the existing generic and desktop configurations. Use temporary scoped
SQLite services, SimpleEmbeddingProvider and MockLLMProvider. This is transport
and lifecycle coverage, not native host acceptance or semantic evaluation.

Preserve the wheel, installed dependencies, live configuration and inactive
pilot files. No model calls, publication, pilot activation or changes to closed
quality and latency results belong in this increment. Prepare a source patch,
review description and concise result receipt with explicit limitations.
