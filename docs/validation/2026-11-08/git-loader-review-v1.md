# Git loader review: integration withheld

The seven-file runtime manifest is not approved for cache integration. Inspection of the installed binaries found LoadLibrary imports in the wrapper, native Git, and each of the five bundled DLLs. Reviewing only the Git source tree does not establish the loading behavior of those bundled libraries. The prior functional manifest check remains valid at its stated scope.

This closes the current source-inspection increment with an unresolved dependency boundary. Do not keep extending it into an unbounded binary provenance investigation. Keep authoritative Git resolution active and perform the already permitted complete-path attribution before deciding whether further cache work is justified by the remaining latency budget. No cache performance benefit has been measured.

## Evidence

The read-only helper `.venv/inspect-git-loader-strings-v1.py` inspected the same seven installed paths as the earlier runtime inspection. It wrote `.venv/git-loader-source-v1/binary-loader-observation.json`, containing SHA256 bindings, loader imports and bounded ASCII/UTF-16 DLL string literals. No inspected binary was executed by this helper. A string is not proof of a reachable loader call; absence of a string is not proof that a library cannot be loaded.

Native Git imports LoadLibraryA, LoadLibraryExA and LoadLibraryW. Its literals include bcrypt, kernelbase, ntdll, psapi and ucrtbase. The bundled iconv, intl, pcre2, winpthread and zlib DLLs each import LoadLibraryW. These imports do not by themselves establish unsafe loading or a reachable cache invalidation failure.

In the downloaded Git `v2.51.0.windows.1` source, Makefile line 2600 links native Git using GITLIBS, while lines 1711 and 2957-2970 explicitly attach the lazy curl object to IMAP/HTTP programs. No curl_global_init literal was found in the seven inspected binaries. Together these observations support treating the PATH-based lazy curl code as a separate executable concern; they do not constitute a complete installed-build reachability proof.

The [Windows loader documentation](https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order) distinguishes loaded modules, KnownDLLs and subsequent filesystem search locations. Thus an ordinary LoadLibrary call is not automatically evidence that a mutable PATH entry determines the result. Conversely, the System32 restriction in Git's lazyload helper cannot be generalized to every imported loader call.

The source archive and wrapper hashes are retained in [the home-selection regression report](git-home-selection-v1.md). That review already produced a reproduced correctness repair, commit `7065ebaf`, with 53 passing tests. This follow-up changes the integration decision and investigation order, not source code or performance claims.

## Next bounded measurement

Use the current canonical private wheel, not the older queued-capture package referenced by the previous diagnostic helper. Verify installed package hashes before and after the run. Freeze the exact helper, native launcher, isolated settings, dependencies and schedule before any measured operation. Include launch, transport, authoritative Git, authorization, durable append, response handling and host-visible command completion. Preserve nested intervals separately and correlate every acknowledgement with reopened durable state and completed jobs.

The old helper targets a different artifact and JSON outbox, so it must not be run unchanged. Adapt its tracing to the current SQLite queue and response-completion behavior first. The one permitted attribution run and later paired comparison remain unrun. Neither a source inspection nor diagnostic preparation consumes an acceptance case or starts the original pilot.
