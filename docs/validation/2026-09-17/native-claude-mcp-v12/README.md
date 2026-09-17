# Native Claude MCP read contract

The v12 run used the verified source commit `6600dc2` and wheel SHA256
`349737206735f15b59562f3c7d53af51464176473b5de1fa12d2783cf039a25a`.
The plugin ran in a disposable project with an explicit profile and local
SQLite store.

This run observed the actual plugin MCP connection and two native calls in
order: `mnemos_health`, then `mnemos_retrieve`. The result IDs matched their
calls. Health reported the running fixture, its local database, capabilities,
profile, and project. Retrieve returned the seeded canary in a fact whose own
scope was `project`.

The run also observed all three configured hook events, durable restart, and
the reconstructed Stop replay. The historical exact-two-native-captures gate
remains a separate failed diagnostic because the hook stream includes four
normal events and one provisional event. The newer identity contract accepts
multiple positively associated assistant transcript events and keeps the
provisional event extraction-ineligible.

This is a bounded transport and persistence observation. The fixture uses
constant vectors and an unavailable cognition provider, so it makes no
retrieval-quality, model, or semantic-memory claim. The v11 plan-mode result
and v12 default-mode result suggest permission mode affected tool dispatch,
but these two runs do not prove that it was the exclusive cause. The run used
no handmade MCP settings and no approval bypass.

The private fixture, raw native stream, transcript, and authentication material
are excluded. The copied report and runner are byte-identical to the executed
v12 files. The source was clean and pinned before and after the native run, and
cleanup removed the copied authentication file and runtime descriptor.
