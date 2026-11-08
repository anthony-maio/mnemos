# Legacy runtime coverage and plugin interpreter repair

Runtime testing found and repaired a Windows plugin-launcher failure. The
original 0.6.2 source passes 343 of 344 tests on Windows Python 3.10 and 3.11;
its plugin-manifest roundtrip fails because bare `python` selects the base
interpreter without the installed dependencies. The separate repair at
`a229d8fd` passes all 349 source tests in each of five environments.

The existing memory wheel is unchanged. Its 41 installed storage and recency
regressions pass in every environment, including the original Windows runs
with the launcher failure. All 35 installed package files match the wheel
before and after testing. The repair changes the source plugin wrapper, its
tests and one explanatory document; it does not change the memory package.

| Environment | Python | Original source at ace62950 | Repaired source at a229d8fd | Installed regressions |
| --- | --- | --- | --- | --- |
| Windows x64 | 3.10.18 | 343 passed, 1 failed | 349 passed | 41 passed |
| Windows x64 | 3.11.13 | 343 passed, 1 failed | 349 passed | 41 passed |
| Debian 13, WSL 2 x64 | 3.10.21 | 344 passed | 349 passed | 41 passed |
| Debian 13, WSL 2 x64 | 3.11.15 | 344 passed | 349 passed | 41 passed |
| Debian 13, WSL 2 x64 | 3.12.13 | 344 passed | 349 passed | 41 passed |

There are no skipped cases in these runs. The original 344 test identifiers
match the earlier source suite. The five additions exercise command lookup,
missing executables, explicit relative paths, exclusion of an unlisted current
directory and the actual child interpreter. They add to the existing explicit
absolute-path test. A new subprocess regression reproduces the escape from the
Windows virtual environment before the repair.

The wrapper now resolves a command against the configured PATH before starting
the MCP server. It handles Windows executable extensions and preserves the
virtual environment's interpreter symlink on Linux. Explicit paths remain
literal, and the default managed-environment path is unchanged. A missing
command produces an error before launch. Strict typing for the wrapper and
formatting for the two Python files pass.

The [protocol](../../plans/2026-11-07-legacy-platform-v1.md) binds the original
source and existing wheel before the checks. Each environment starts from a
fresh temporary installation. The source suite uses an exact committed-source
export; installed regressions run from copies outside that source tree. Module
origins and source/archive/installed bytes are audited. The repair uses new
source copies in the same environments, preserving the original results and
dependency sets. The unchanged installed regression suite is not repeated.

The tests include MCP subprocess roundtrips through the checked-in generic,
desktop and plugin configurations. They do not run the native Claude or Codex
applications. External network connections are disabled during tests; dependency
installation uses package indexes. CLI startup and pip check pass in all five
environments. No production benchmark, fresh semantic questions or hosted model
evaluation is part of this increment.

Python 3.10 cannot use seven versions pinned in the earlier Python 3.12
validation environment. Their package metadata requires Python 3.11 or newer.
The corrected setup retains compatible pins and freezes these replacements
within mnemos's unchanged dependency requirements:

| Dependency | Earlier pin | Python 3.10 pin |
| --- | --- | --- |
| platformdirs | 4.14.5 | 4.12.4 |
| pycparser | 3.12 | 3.11 |
| pydantic-settings | 2.16.2 | 2.15.0 |
| python-multipart | 0.0.33 | 0.0.32 |
| rpds-py | 2026.9.1 | 0.30.0 |
| sse-starlette | 3.6.0 | 3.5.1 |
| starlette | 1.10.0 | 1.7.0 |

Conditional dependencies and Windows-only colorama/pywin32 are recorded in the
resolver reports and receipt. This is a test-environment correction, not a
change to the package's declared requirements.

Four failed setup attempts are retained. The first Windows attempt used a
relative bundle path after changing working directory; the WSL attempt rejected
an incompatible pin. A later runner assertion missed pywin32 as a Windows-only
dependency, and the command sequence continued into another failed installation
per platform with the old pins. None of these attempts ran source or installed
tests. The first wrapper prototype also exposed Python 3.10's executable-suffix
behavior; a subsequent test assertion needed platform-aware path comparison.
These failures remain separate from the final five passing source runs.

The [receipt](legacy-platform-v1-receipt.json) records revisions, dependency sets,
counts and exact artifact bindings. A three-file review patch reconstructs the
repaired source from `ace62950`, with an offline PR draft. The 170-member private
archive has SHA-256
`cd9341752082b99fef203af518330bc923dfcf52c050f3f6d45bf7642d7f5f5c`.

The earlier Windows 3.12 package evidence keeps its original binding. These
new WSL checks do not establish native Linux CI, macOS, other architectures or
later Python support. The frozen functional plugin also contains a bare-command
override path; this report validates only the separate legacy repair, leaving
that integration for a scoped follow-up.

The original wheel and source archive, chronology results, failed production
gates, canonical delivery and fourteen watched live paths are unchanged. The
repaired source has not been packaged for publication or activated. Native
application acceptance, release qualification and actual pilot use remain open.
