# Windows private ACL repair

Source based on `e7319e05525755e94ee945b9ee59a38295b05977` plus `source.patch` and the archived
new test file passed 1,039 tests with 4 skips, Black, strict
mypy, and JavaScript syntax. All nine real Windows permission regressions ran.
Input hashes were stable across validation; fixtures were outside Git.

The permission helper previously passed `0x10000004`, which did not protect the
DACL against inheritance. It now passes `0x80000004`. Files receive one explicit
current-user full-control rule; directories receive the same rule with child
inheritance enabled while their own DACL is protected. The tests inspect actual
Windows permissions and verify that changes to the parent do not widen access.
Microsoft documents [protected DACL behavior](https://learn.microsoft.com/en-us/windows/win32/secauthz/security-information)
and publishes the [numeric flags](https://github.com/microsoft/GuardedFabricTools/blob/master/CertificateManagement/CertificateManagement.psm1).

Before writing permissions, the helper now reads the current security descriptor.
It skips the write only for a present, protected DACL whose bytes exactly match
the required current-user policy. There is no path cache. Inherited permissions,
extra rules, a missing descriptor, or failed descriptor/control/ACL queries take
the repair path. Native descriptor allocations are freed even when comparison
fails. Tests verify that a later extra Everyone rule is removed and that an
unchanged private ACL avoids another permission write.

The [native boundary observation](../native-codex-stop-boundary-v1/README.md)
and [native profile](../native-codex-stop-profile-v2/README.md) used the older
installed `5af3d15` wheel. Both completed; the profile attributed about 1.175
seconds to permission updates. Its lock file still had an inherited rule.
Those runs located the issue but do not validate this newer source in a native
host. The small sequential filesystem microprobe did not reproduce the native
cost and is not a controlled speedup estimate. No latency gate is promoted.

Negative attempts are retained. The initial three regressions failed on actual
unprotected ACLs; two later regressions failed on redundant native permission
writes. Early PowerShell fixture problems and a fault-injection setup that
short-circuited before its target API are preserved separately. The initial full
run failed that last test setup. The final fixture first establishes a protected
ACL and then adds an extra rule so the ACL query failure is exercised directly.
Neither permission requirements nor performance deadlines were relaxed.

This is source validation. A fresh installed package and native run for this
repair remain required. The frozen Ollama evaluation and live host configuration
were not changed.
