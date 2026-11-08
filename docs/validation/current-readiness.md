# Development candidate readiness

Updated November 8, 2026. This is a source and validation summary, not a release
or pilot-completion announcement. [PyPI](https://pypi.org/project/mnemos-memory/)
still serves 0.6.0; the local-memory candidate is unpublished 0.7.0b1.

## Runtime and installation

The latest private wheel contains runtime source
`4d2d5c5333ed230bcb59f917c3a03e674a41a35c`. Its SHA-256 is
`d86b925528bf8cd6eb8f703d2b63730a7595eaf8e982261c7038c324aa64ed5e`.
The isolated installation verified 99 package files and 57 pinned dependency
versions and passed real FastEmbed CLI/MCP restart recall. The surrounding
forget, transport, paging and suppression suite passed 105 tests. These checks
do not establish semantic quality or native-host lifecycle completion.

The README's source-install commands pin that exact commit, avoiding drift in
the development branch. A source install still resolves dependencies; it does
not reproduce the private environment's exact dependency inventory. Existing
pilot environments are separate and are not silently refreshed.

The independent validation runner at `bee4f320` passed 4,545 tests with 89 skips.
That suite belongs to the improvement worktree; it is not a full-suite result
for the distinct runtime checkout.

## Native workflow

The [Codex v12 lifecycle](2026-11-07/native-codex-v12.md) passed its six phases
against `f5a944ea`, with hooks disabled. Its pass remains bound to that artifact.

The [latest Claude v8 attempt](https://github.com/anthony-maio/mnemos/blob/c5cb60ea4e192fe9d91b0bf55bdfc1196e147efb/docs/validation/2026-11-08/native-claude-record-pages-v8.md)
used the latest private runtime package and failed the 180-second forget deadline.
The first four phases passed. Inspection reached nine of eighteen candidates;
no deletion occurred. Final suppression, original-source replay and fresh-host
recall were not attempted. Cleanup passed. The current complete workflow gate
remains open; Desktop and native Hermes acceptance also remain unverified.

## Semantic quality and timing

No evaluated support configuration meets all required quality and timing gates.
The [retained Qwen3.5 screen](https://github.com/anthony-maio/mnemos/blob/d615d31de7883bb29cb230d9225c089ff3ae0be9/docs/validation/2026-11-08/qwen35-support-screen-v1.md)
completed 176 support requests over 44 known questions across four retrieval
arms. Hybrid precision was 65.85%, correct abstention 9/17, and support-only p95
2,416 ms. It is closed and remains outside production retrieval. Critical
retention and fresh acceptance were not established.

The 500 ms complete warm-retrieval and 100 ms complete-capture gates remain open.
Internal transaction or model-only timings do not establish those boundaries.
Historical results and their separate artifact bindings remain in the
[development evidence log](development-history-through-2026-11-08.md).

## Daily use and publication

The separately approved optional explicit-memory pilot was activated. Its setup
does not constitute the original two-week daily-use pilot, whose entry gates
and qualifying days remain incomplete. No actual-use record should be inferred
from elapsed time or a synthetic test.

The README, source metadata, private wheel, published package and website
examples represent different delivery states. A documentation change does not
publish a package or update a live site. The [release checklist](../release-checklist.md)
still controls publication; none of the open requirements is waived here.

The live [restart example](https://mnemos.making-minds.ai/examples/restart-recall/)
still pins its earlier `7d5fc277` source and 18-tool receipt. The private candidate
above has 19 tools. The live example is valid historical evidence for its own
source and does not validate the newer candidate.
