# Explicit Codex runtime and storage settings

The generated Codex MCP and command-hook configuration now accepts explicitly
supplied `MNEMOS_AUTO_START`, `MNEMOS_RUNTIME_DIR`, `MNEMOS_STORE_TYPE`, and
`MNEMOS_SQLITE_PATH`. Disposable native phases pass all four settings to the
generator. They no longer rely solely on subprocess environment inheritance.
The existing allowlist still rejects unrelated keys and does not copy ambient
credentials or arbitrary environment values.

Two regression checks failed before this repair: the generator rejected the
new explicit keys, and disposable configuration omitted them. The focused run
now passes 93 tests with two platform skips; 11 documentation checks pass.
The tests parse the MCP environment and decode every generated hook bootstrap.
They cover generated settings, not a newly observed native Codex outage.
Full package checks and a separately frozen native run remain required.

The index binds the original JUnit bytes and implementation diff. Existing
Claude outage evidence remains bound to its earlier installed package.
