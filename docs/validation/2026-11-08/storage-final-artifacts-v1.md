# Current patch artifacts and Desktop isolation

PR #9 remains draft/open at `b51b06c620468155ba93ef36564816a600925705`.
All eight jobs in [CI run 42512165698](https://github.com/anthony-maio/mnemos/actions/runs/42512165698)
completed successfully: Python 3.10, 3.11 and 3.12 tests, lint/types, benchmark,
Tier 1 and Tier 2 MCP smoke checks, and release readiness. Generic-client smoke
tests do not establish actual Desktop acceptance.

The clean current source was rebuilt with Hatchling 1.33.0 in an isolated build
environment. Both artifacts pass Twine. The wheel is byte-for-byte identical to
the artifact used by the recorded installed and native checks. All 35 runtime
files match source, wheel and sdist. The new sdist differs from the earlier one
only in CHANGELOG.md, docs/client-compatibility-matrix.md and
docs/release-checklist.md. These are the expected documentation corrections.
No unchanged runtime/native matrix was rerun.

| Artifact | SHA256 |
| --- | --- |
| mnemos_memory-0.6.1-py3-none-any.whl | a3abff76729ea4cc0fa0e8fbba8a087d60e50b8fb5c2324e6222ff44e6ff4463 |
| mnemos_memory-0.6.1.tar.gz | 3d017b8e842f4e93a94977392f81cf7f2fad1b5b2ba22d19a535b6f3a776da02 |

Private artifacts, build logs, dependency versions and parity result are retained
in `.venv/storage-final-docs-v1`. The previous sdist and all prior receipts remain
unchanged. The PR description now records the current artifacts, native plugin
result and remaining requirements. Nothing was merged, tagged or published.

## Desktop isolation finding

The installed Windows MSIX package is Claude 4.23118.0.0. Read-only inspection
of its ASAR startup code confirms a CLAUDE_USER_DATA_DIR override. The inspected
index.pre.js SHA256 is
`53efd191a6243a69bb400d19258706bfd03815c811509d1b688b2c0755d5f2c0`.
Profile redirection alone is insufficient to preserve normal host configuration:
review found that startup invokes an MSIX Open-with registration routine after
acquiring the primary-instance lock. That routine writes HKCU registrations or
deletes them when the execution alias is absent. Redirecting LOCALAPPDATA can
make the alias lookup absent. The profile override does not guard this path.
The reviewed main startup chunk is index.chunk-CUXoxsyW.js, SHA256
`8a49065ed8437a1e4d91c972524044ad58c73eebcac521729a9bc9d9472fab32`.
Review also found startup browser NativeMessagingHosts registration changes;
the apparent skip flag requires developer-approved test hooks in this build.

Desktop was not launched and normal registry/profile state was not changed.
Actual Desktop validation needs an isolated OS user or VM, with its own login
and checked registration/storage paths. Do not treat a generic MCP client,
separate Electron profile or Claude Code session as Desktop acceptance.
WindowsSandbox.exe and the checked VMware/VirtualBox command-line launchers are
not available on this machine. This is not proof that no VM exists elsewhere.

Desktop acceptance, daily-use pilot completion and any final publication
authorization remain outstanding. The six-priority goal also retains its
semantic, complete-latency and current functional-delivery obligations.

## November 12 delivery refresh

The [installation/rollback handoff](../../releases/v0.6.1-handoff.md) now collects
the existing exact artifacts and open release requirements. PR #9 is still
draft/open at the same head; all eight CI jobs still show success. Fresh
SHA-256 checks match both artifact hashes above. PyPI's current JSON metadata
and GitHub release list still identify 0.6.0 as the published version; 0.6.1
is absent. No package or native matrix was repeated.

Installed Claude Desktop is now 5.1231.2.0. Read-only inspection finds that its
packaged startup rejects CLAUDE_USER_DATA_DIR without a developer-signed,
path-bound authorization token. The native-host-install skip flag requires
the same developer-approved gate. Native-host synchronization still contains
direct registry write/delete calls, and the app manifest disables registry
write virtualization and excludes the relevant NativeMessagingHosts keys.
The previous build's specific OpenWith finding was not established for this
new build and is not transferred to it.

No safe environment-only launch is proven. Desktop was not launched. Current
Hyper-V enumeration is denied by the host authorization policy; the checked
Sandbox/VM launch commands remain unavailable. A separate Windows environment
choice is pending from the user. The new read-only inspection receipt is
retained privately in the existing artifact packet as
`desktop-isolation-51231-review.json`, SHA-256
`eba8ca90afc4b0af1bb63df449a79408345181e35eedcb7882124c6a8612583f`.
Normal registry, profile, credentials and databases remain unchanged.
