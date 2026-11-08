# Legacy candidate runtime coverage

Verify the existing unpublished 0.6.2 artifact on five additional environments:
Windows CPython 3.10.18 and 3.11.13, and Debian 13 under WSL 2 with CPython
3.10.21, 3.11.15 and 3.12.13. Use the already installed interpreters and fresh
temporary virtual environments. These cells supplement the recorded Windows
3.12.11 evidence; they do not repeat it or establish native Linux CI, macOS,
other architectures or later Python support.

Bind source `ace6295030c78850373404086748696f4fad5f5e` and the existing wheel
SHA-256 `4f34b8524ec5352e66a554864cf26f624393afe64834004a00fcd18cbf4d84a3`.
Export the committed source into disposable directories. Preserve every test
file and the candidate's configuration. Do not rebuild or change the wheel.

Install the wheel with its MCP extra, pytest and pytest-asyncio under the
existing package-validation constraints. Record exact resolved dependencies,
including dependencies selected only by a platform or Python-version marker.
Do not silently relax incompatible constraints. An installation failure is a
recorded result requiring diagnosis before any revised attempt.

In each new cell, run the complete source test suite, then the 41 installed
storage and content-recency regressions from copies outside the source tree.
Verify module origins for both runs and compare all 35 installed package files
against the wheel before and after testing. Run installed CLI startup and pip
check. Retain logs, test identifiers, failures, skips and dependency manifests.
The existing MCP manifest tests exercise subprocess harnesses; passing them
does not establish native Claude or Codex application acceptance.

Use temporary configuration and database paths. Clear inherited provider
settings and disable external socket connections during tests while allowing
loopback and stdio. Dependency installation may access package indexes. No
hosted model, fresh semantic questions, production benchmark, live host
configuration, database migration in place, publication or pilot activation
belongs in this increment. Preserve the earlier failed benchmark and all
closed chronology/support/timing results.

Run each cell once. Diagnose concrete failures and preserve the first attempt
before making any necessary harness correction. Do not rerun passing cells or
edit tests to hide a platform failure. Finish with a result table that names
the actual operating systems and patch versions, plus exact source/artifact
bindings and the still-open release requirements.
