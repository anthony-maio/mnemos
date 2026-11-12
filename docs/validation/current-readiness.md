# Development candidate readiness

Updated November 11, 2026. This is a source and validation summary, not a release
or pilot-completion announcement. The public package and website observations
below were last checked November 8. [PyPI](https://pypi.org/project/mnemos-memory/)
served 0.6.0 then; the local-memory candidate remains unpublished 0.7.0b1.

## Runtime and installation

Recall repair `73d4a242b27b9ff34da9bda4a1db4545045e5e65`
[closes a service-boundary recall race](https://github.com/anthony-maio/mnemos/blob/cd46bc53/docs/validation/2026-11-11/recall-snapshot-v1.md):
deleted, corrected or newly ineligible selections cannot reuse an earlier
retrieval result during response assembly. SQLite responses read facts,
evidence and source status from one assembly snapshot, and filtered context
is recounted. All 178 targeted checks pass, including 13 new regressions;
focused typing and formatting pass. The new private package and native result
below include this repair. It is not a semantic or timing pass.

Earlier runtime source `f1314f6d40477d59f6a31ecbc904c3e7a3ce5c7f` fixes a
reproduced inspection race during concurrent correction/forget. Each SQLite
inspection now reads its fact, evidence, source status and revisions from one
snapshot. 116 distinct relevant checks pass. Two pre-existing mypy errors in
forget-preview code remain. The fix is now privately packaged and installed,
with five installed inspection-race tests passing. It does not explain or fix
the earlier Claude timeout.

The [latest private wheel](https://github.com/anthony-maio/mnemos/blob/17f561a0/docs/validation/2026-11-11/native-codex-v14.md)
contains source `9c326cbc239e41581b7754b1035bb897c281e42a`, including both
repairs. Its SHA-256 is
`849b0d226596f8a809d108cd860caf3229c5f168c81b2d4b23faf66dd87cbd29`.
All 99 installed files match source. The separate installation reuses the same
57 dependency versions, with 4,243 copied dependency file hashes verified and
pip check passing. Eighteen installed regressions and real FastEmbed CLI/MCP
restart recall pass. This is controlled dependency reuse, not a fresh dependency
resolution or cross-platform installation claim. No semantic quality follows.

The README's source-install commands pin that exact commit, avoiding drift in
the development branch. A source install still resolves dependencies; it does
not reproduce the private environment's exact dependency inventory. Existing
pilot environments are separate and are not silently refreshed.

The independent validation runner at `bee4f320` passed 4,545 tests with 89 skips.
That suite belongs to the improvement worktree; it is not a full-suite result
for the distinct runtime checkout.

## Native workflow

The November 11 [Codex v14 attempt](https://github.com/anthony-maio/mnemos/blob/17f561a0/docs/validation/2026-11-11/native-codex-v14.md)
passed all six explicit-memory phases against
the current `9c326cbc` wheel: remember, restart recall, same-ID correction,
corrected recall, reviewed forgetting, and final restart recall after original
source replay suppression. Six fresh sessions produced 19 correlated MCP calls.
All six complete tool inventories were verified. Hooks and Jev were disabled;
the normal hosted Codex model received synthetic scenario text. Memory processing
and embeddings stayed local. This does not establish offline model inference,
normal hook trust, the full pilot configuration, quality or latency.
The offline audit reconstructed all 19 calls and verified the original prompts,
inventories, installed guidance, replay identity and final database state. This
forget selected only the target; multi-fact selection and batch review are not
covered by the native pass. The bounded preflight review cleared wrapper changes
only. Normal files, 4,361 bound inputs and embedding assets matched preflight
hashes at cleanup, with credential copies removed and zero owned survivors.

The earlier [Codex v13 lifecycle](https://github.com/anthony-maio/mnemos/blob/41b371a99aca05ca04a3418d04e6c229ccafce8c/docs/validation/2026-11-11/native-codex-v13.md)
remains a separate passing result on `f1314f6d`.

The earlier [Codex v12 lifecycle](2026-11-07/native-codex-v12.md) passed against
`f5a944ea`; its original evidence remains bound to that artifact.

The [latest Claude v8 attempt](https://github.com/anthony-maio/mnemos/blob/c5cb60ea4e192fe9d91b0bf55bdfc1196e147efb/docs/validation/2026-11-08/native-claude-record-pages-v8.md)
used the older `4d2d5c53` private runtime package and failed the 180-second forget deadline.
The first four phases passed. Inspection reached nine of eighteen candidates;
no deletion occurred. Final suppression, original-source replay and fresh-host
recall were not attempted. Cleanup passed. That failure is preserved; the current
Codex pass does not transfer to Claude, Desktop or native Hermes acceptance.

## Semantic quality and timing

No evaluated support configuration meets all required quality and timing gates.
The November 11 direction requires local operation with optional Jev. The
[one optional Jev comparison](https://github.com/anthony-maio/mnemos/blob/d5af6d34/docs/validation/2026-11-11/jev-support-screen-v1.md)
completed 44 known hybrid questions with 100% selected precision and no-answer
abstention, but only 21/27 retention/recall. Its 441.73 ms p95 covers support
only. It fails the original gates and does not establish local support quality.
No default or live configuration changed, and there is no automatic cloud fallback.

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
