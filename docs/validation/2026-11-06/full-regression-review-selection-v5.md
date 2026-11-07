# Full regression for review selection v5

The frozen suite at `15229fb75c0520966e9a7124fcb6efc4a9d9c865` passed
2,820 tests, with four skips, in 501.77 seconds. There were no failures or
setup errors. The [receipt](full-regression-review-selection-v5.json) records
the terminal counts, skip reasons, source identity and private archive hashes.

The run explicitly selected canonical website delivery source `08d393ee`.
Its bytes, all 208 restored historical fixture files and the tracked source
remained unchanged. It used the isolated Python environment and a fresh
external temporary directory. Two skips require the installed WSL Hermes
Python; two require a POSIX executable and shell.

The private archive contains the log, JUnit report, control record, runner,
archiver and restored-fixture manifest. All six members verified. Its SHA-256
is `38d4eb85104d0dfc9e9e9e119a3c31884b92ea6993af78fb74d0237e1c7003e9`.
Earlier failed runs remain unchanged. This regression does not measure memory
quality, establish pilot readiness or imply publication.
