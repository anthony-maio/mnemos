# Windows Git home selection invalidation

The inactive policy-owner prototype now observes the filesystem input used by the Windows Git wrapper to choose HOME. Implementation commit: `7065ebaf` in the capture worktree. No live installation, memory database, host configuration or cache integration changed.

When HOME is unset, the wrapper checks whether HOMEDRIVE plus HOMEPATH names a directory. If it does, that becomes HOME; otherwise the wrapper falls back to USERPROFILE. The environment can remain unchanged while creating the first directory switches the global configuration path. Observing only the previously discovered configuration paths misses that switch.

The regression used disposable directories and the installed Git executable. It first established a cache entry, created the alternate home with a malformed `.gitconfig`, and verified that authoritative Git failed. Before the fix, cached resolution incorrectly succeeded. The retained red run has one failure. After the fix, the cache invalidates and propagates the authoritative failure. A separate check covers directory creation, removal, explicit HOME bypass and rejection of relative fallback paths.

The owner observes the candidate and its parents using the existing content/security observation helper. Missing paths are recorded; reparse points and unsupported paths force authoritative fallback. These observations are sampled with the rest of the policy before and after dependency discovery. This does not claim an atomic filesystem snapshot.

Validation: 53 tests passed in 44.40 seconds across policy owner, pinned runtime, policy probe, configuration dependencies and project identity cache. Private evidence is retained at `.venv/home-policy-red-v1.xml` and `.venv/home-policy-green-v1.xml` in the improvements worktree. `git diff --check` passed before commit.

## Runtime source review and remaining limit

The [official wrapper documentation](https://gitforwindows.org/git-wrapper.html) links to [wrapper source at commit 0a7407a](https://github.com/git-for-windows/MINGW-packages/blob/0a7407a39c3015cc7a3c296d8a0db38439c65eed/mingw-w64-git/git-wrapper.c). Its lines 132-163 explain the home-selection behavior, reproduced against this installation above. That linked source revision is not asserted to be the provenance of the installed wrapper binary.

The Git source archive for tag `v2.51.0.windows.1` was downloaded from GitHub into `.venv/git-loader-source-v1/source.tar.gz` and inspected without execution. SHA256: `1e389ecc31462e2d1faf37f6a50ec583797dfc212f5b5c7f4779c57b7f92679c`. The saved linked wrapper source is `.venv/git-loader-source-v1/git-wrapper.c`, SHA256 `52cea50c2292e7200a9e6e701f4c48f60bcfe7be519ba42b325fe035b2683eb9`.

The source review does not establish a complete runtime dependency closure:

- `compat/win32/lazyload.h:49` restricts its helper to System32. That restriction applies to that helper, not every loader call.
- `compat/mimalloc/prim/windows/prim.c` contains ordinary LoadLibrary calls for several Windows DLLs. The Windows build configuration enables mimalloc. Compile-time branches and actual paths still matter.
- `compat/lazyload-curl.c:43-80` contains PATH-based dynamic loading. Finding that source does not prove it is linked into or reached by the identity/configuration commands.
- The seven-file observed manifest and two directory inventories therefore remain functional test inputs, not an approved complete runtime binding.

Next: resolve the loader/build reachability boundary before cache integration. The bounded complete-path attribution and paired comparison in the convergence reset remain unrun. This correctness repair makes no latency claim and does not consume either timing run.
