# Delivery review preparation

A private review bundle now reconstructs the tested `749f1213` source tree from
the preserved delivery base `9770c009`. It contains the complete binary-capable
patch and the already validated wheel and source archive. Applying that patch to
a private Git index produced the exact target tree. The real worktree indexes,
branch heads, delivery checkout and website were unchanged.

This is preparation for integration. It is not an applied update, a new package
validation, a refreshed public-branch observation or a release approval.

## What differs from the delivery base

The complete patch changes 296 tracked files, including tests, benchmark helpers,
protocols and validation records. Twenty-five changed files are product code or
the README. Package version metadata is unchanged at `0.7.0b1`.

| Area | Implemented behavior | Evidence and limits |
| --- | --- | --- |
| Graph persistence | Missing sources are checked under the write transaction; neighbor replacement preserves caller work and rolls back partial changes | Storage regressions and the separate public-branch backport remain bound to their recorded sources |
| Explicit-memory workflow | MCP instructions distinguish queued jobs from completed jobs; bounded, lossless forget-review pages retain source selections and stale-preview checks | Codex and Claude native lifecycle passes apply to `6d0e5ba6`; the newer wheel does not inherit those passes |
| Defaults and setup | All five experimental modules default off; explicit memory remains available; ordinary settings edits preserve module choices | Existing explicit settings still apply; omitted flags adopt the new defaults, so this is a behavior change |
| Context delivery | Optional identified tokenizer with byte fallback; full qualified records; stale revisions omitted; opt-in compact MCP responses | Full responses remain the default; native compact measurements are bounded checks and do not establish a causal host-token saving |
| Experimental support and extraction | Typed support judgments, reconciliation guards and several model-comparison helpers | Their negative evaluations remain recorded; no runtime call site was found for the new answer-support checker or target-identity comparator |

The product sources in the current improvement checkout still match `749f1213`.
Later changes are benchmark and documentation work. The review bundle deliberately
targets that exact validated tree: it does not include the later accepted-corpus
runner, fresh comparison or current README edits. Final integration must include
the final documentation and evaluation records, then bind any newly built
artifacts to their actual source. The bundle is not the final release candidate.

## Exact preparation evidence

- Base commit: `9770c009f8d3b4919ecfd18810d700badc53c638`.
- Target commit: `749f1213381848a0d6040bf8a241829b18b655a1`.
- Reconstructed target tree: `df99c070be48392c857ed0258805d52630f7ca0c`.
- Patch SHA-256: `97091a0be1acf7c2291c115639e28885b891e3106d343d3b2393af0b061162ee`.
- Verified private archive: five members, 2,917,859 bytes.
- Archive SHA-256: `4f035c54959afa5617f8af214ca95e4a079416e889a6b954f7c9525c069416fe`.

The wheel and source archive hashes match the earlier
[package checkpoint](checkpoint-749f1213.md). That checkpoint passed 3,221 tests
with four platform-specific skips, checked both package artifacts and completed
an installed CLI/MCP restart check with real embeddings. This preparation only
rechecked those artifact bytes and patch application; it did not rerun that suite
or transfer its claims to later code.

The README now names that package checkpoint, the newer negative QA development
result and the in-progress fresh retrieval-only comparison. Its twelve existing
documentation consistency checks pass with the explicitly selected website
delivery source. The website remains at its preserved `08d393ee` checkout, and
its installation example retains its original source binding.

Fresh retrieval evaluation is still running. Answer-support acceptance, the
full-module comparison, remaining native/scale checks, final delivery alignment
and actual two-week pilot remain open. The pilot remains on its earlier
installation and is inactive at zero days. No live configuration, public branch,
package release or hosted page changed.

- [Preparation receipt](delivery-review-v1-receipt.json)
- [Fresh comparison status](fresh-accepted-v1.md)
- [Earlier delivery inventory](../2026-11-06/delivery-state.md)
- [Experimental defaults and upgrade behavior](../../experimental-modules.md)
