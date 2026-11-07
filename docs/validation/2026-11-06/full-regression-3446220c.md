# Full regression checkpoint

The full suite at `3446220cf48d03d24df06b292d9e8aefe66d177e` passed with
2,724 tests passed, four skipped, no failures and no errors. The pytest summary
reported 526.70 seconds. The [receipt](full-regression-3446220c.json) binds the
source commit, environment versions, JUnit XML, log and verified private archive.

Two skips require an explicitly configured installed WSL Hermes Python. Two
require a POSIX interpreter and shell. They remain unrun; the Windows suite
does not establish native WSL Hermes or POSIX execution.

An earlier attempt placed pytest's temporary directory inside this checkout.
That made independent fixture directories inherit the same Git project root.
An isolated probe reproduced the collision and confirmed distinct identities
outside the checkout. The attempt was stopped after 549 visible test marks,
including six failures, and retained as an aborted setup attempt. It is not
counted as product regression evidence. The complete corrected run used a fresh
external temporary directory and left the source unchanged.

The archive retains both attempts' logs and run metadata, the corrected JUnit
results and the scope probe. Its nine members were read back and verified;
SHA-256 is `79401f7d75259c1475d68b1f36cff0533a40bc88a21522062a76fc312c593cac`.
No historical scores, live memory or host configuration were changed.

These tests cover source contracts and regressions. Retrieval quality, native
host acceptance, installed artifacts, pilot use and publication have separate
evidence and cannot be inferred from this pass count.
