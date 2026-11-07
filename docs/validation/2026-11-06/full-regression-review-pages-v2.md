# Full regression after restoring archived fixtures

Source `7f5aec4b587599ccf0592af236cc5b8d0fc2a5eb` passed 2,806 tests with
four platform-specific skips in 471.40 seconds. There were no failures or
errors. The run used a fresh temporary directory outside the checkout, and
tracked source stayed unchanged throughout.

Two skips require an installed WSL Hermes interpreter. Two require a POSIX
executable and shell. They remain untested on this Windows run.

The [receipt](full-regression-review-pages-v2.json) binds the terminal control
record, full output and JUnit result. All four private archive members verified;
the archive SHA-256 is
`4a3d5cadab4ad306f20f974c77ae8456bbf449cafae9cac8fb46cba0c5164419`.

The [earlier run](full-regression-5de1844f-missing-fixtures.json) remains failed:
two historical fixture directories were absent. Their 208 files were restored
without changing Git bytes. That run's 2,749 passes, four skips, two failures
and 19 setup errors are preserved separately. This fresh result covers the
paged-review product, independent native verifier and committed v4 protocol.

This is source regression evidence. Installed checks, native host results,
quality evaluation, actual pilot use and publication retain their own evidence
and version bindings.

A later explicit check against the canonical website checkout found that the
website test still required the older `Starter SQLite` label. The full suite's
pass did not establish consistency with that canonical page. The repaired test
requires an explicit website path and checks its actual install commands,
SQLite setup guidance and separation of published and development results.
All twelve documentation checks then passed; formatting and strict types passed.
The [website-check receipt](canonical-website-check.json) preserves both results.
This test-only update postdates the full run and does not change product code.
