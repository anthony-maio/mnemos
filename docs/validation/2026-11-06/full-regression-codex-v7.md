# Full regression before Codex v7

The full suite at `6c12ff94ddf3bc8c15ec3a1c211f8791cba61916` passed
2,845 tests with four platform-specific skips in 502.66 seconds. There were
no failures or errors among 2,849 collected tests. Two skips require an
installed WSL Hermes Python; two require a POSIX executable and shell.

The run used a fresh external temporary directory and explicitly selected
the canonical website checkout. HEAD, tracked source files, the website and
all 208 restored historical fixtures stayed unchanged. The
[receipt](full-regression-codex-v7.json) independently matches the JUnit counts
and binds six verified archive members. Archive SHA-256:
`c721b1544d6e00f9228b01fa2f6ee9e916638fa0b298915cd3009fd3ab93383c`.

This covers the current source, including the optional three-check protocol
and the Codex v7 input binding. It does not establish model quality or installed
acceptance for every source change. Native Codex v7 uses the earlier verified
`6d0e5ba6` wheel; its separate receipt supplies native evidence. Earlier failed
and passing regression runs remain unchanged.
