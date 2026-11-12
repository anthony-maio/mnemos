# Installed recall repair and Codex lifecycle: passed

Canonical commit `9c326cbc239e41581b7754b1035bb897c281e42a`, containing the
recall assembly repair `73d4a242`, now has a qualified private wheel and a
passing six-phase native Codex lifecycle. This closes the changed-artifact
gap left by the source-only repair. It is not a release or full-goal pass.

The wheel and sdist were built from a committed Git archive using the same
Hatchling 1.33.0 backend and passed Twine. All 99 packaged runtime files match
source and installation. Compared with the previous f1314f6d wheel, the only
changed packaged runtime file is `mnemos/memory_service.py`.

The updated README source-download commands were also executed in a fresh
checkout. All 99 packaged source files matched the wheel byte for byte. That
check did not install dependencies or rerun the example; the separate installed
smoke and native attempt provide those results.

The new environment is outside both repositories and every pilot installation.
Its 57 dependency versions match the previous accepted installation. Dependency
files were copied into a fresh empty virtual environment, with 4,243 copied
file hashes checked; pip installed the new wheel without resolving dependencies
and pip check passed. This is controlled dependency reuse, not an independent
fresh dependency-resolution or cross-platform installation result. Eighteen
installed-module regressions passed: 13 recall cases and five inspection cases.
A separate test driver supplied pytest without adding it to the runtime.
The real offline BGE CLI/MCP restart smoke passed, and all 17 retained embedding
assets and the previous installation remained unchanged.

| Native phase | Correlated calls | Verified result |
| --- | ---: | --- |
| Teach | 3 | Original remember job completed with retained source evidence |
| Restart recall v1 | 2 | Original fact and source inspection matched |
| Correct | 6 | Same fact ID advanced to version two |
| Recall v2 | 2 | Corrected content and evidence matched |
| Forget | 4 | Inspected preview and expected-version target deletion completed |
| Final restart recall | 2 | Original source replay was suppressed; target absent, retained fact recalled |

All six fresh sessions used the unchanged prompts, complete tool inventories,
pinned executable/Code Mode host and original 180-second phase limits. The
single Windows-owned attempt completed in 225.09 seconds, with exit code zero
and zero owned survivors. No phase was retried. Credential copies and runtime
descriptors were removed. All 4,361 bound inputs, normal watched files, activated
optional-pilot project files and embedding assets matched their preflight hashes.

The offline audit reconstructed all 19 calls from original trace files, checked
prompt hashes against the original frozen prompts, verified installed guidance
was present in all six model requests, and independently recomputed lifecycle
assertions and final SQLite state. All three unrelated/retained/foreign controls
remained unchanged; one exact original source replay was suppressed. SQLite
integrity and foreign-key checks passed. No additional inference was used for
the audit. Its first adapted version omitted three recorded call-metadata
fields; the retained corrected version compares those fields too and passes.
That offline audit adjustment did not rerun or change the native attempt.

The bounded preflight review cleared wrapper changes only; it did not run the
package or lifecycle. The runtime repair's earlier source review remains
separate. Raw trace reconstruction, package checks and controller receipts
provide the execution evidence here.

Memory processing and embeddings stayed local, with Jev, enrichment and hooks
disabled. Codex's hosted model received only the synthetic scenario. This is
not offline host inference, normal hook trust, native multi-fact deletion,
atomic batch inspection, semantic quality, complete timing or original-pilot
acceptance. The failed Claude and semantic/timing results remain unchanged.
No live memory, pilot install, normal configuration, remote default branch,
package publication or live website changed.

Wheel SHA-256:
`849b0d226596f8a809d108cd860caf3229c5f168c81b2d4b23faf66dd87cbd29`.
Sdist SHA-256:
`c5440a8bb0800167e21d39a24813f973e563a00ac4bfc63fd88befa07cced0c6`.
Runner commit: `14592c80515a298279ec4f3462605da56f287bfb`.

Private package: `.venv/recall-snapshot-package-v1`. Installation:
`C:/Users/Ant/AppData/Local/Temp/mnemos-recall-snapshot-v1`. Native evidence:
`C:/Users/Ant/AppData/Local/Temp/mnemos-improvement-checks-20261106/native-codex-v14-9c326cbc`.
Private archive: `.venv/native-codex-v14-private.zip`, 944 members and
5,713,289 bytes. SHA-256:
`96baf3659ed3df04f0f2abcd87ad84d911b76ab4c9cff05483844b35890a382e`.
All 941 included file hashes were verified on readback; executable and copied
dependency bytes are referenced by frozen hashes instead of duplicated.
