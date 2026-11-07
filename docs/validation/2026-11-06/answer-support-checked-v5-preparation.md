# Three-check support preparation

Product `e5373791971ec6ade560bcf149f48918d6b3042e` adds the optional
`checked_v4` answer-support protocol. It separately checks the requested value,
own source support and supported comparison conflicts. All three selected
components must meet the threshold. The existing scoped snapshots, explicit
reference date, dependency checks, failure accounting and budgets still apply.
The protocol remains outside production retrieval.

The 24 new contract cases failed before implementation. After implementation,
78 focused checker/comparison tests passed. A new runner-selection case then
failed before its guard was added; the final runner group passed 30 tests,
including overlapping checker cases. Together these cover 79 distinct tests.
Formatting passed for four affected Python files, and strict types passed for
both source modules. Original red logs are preserved.

Actual request construction over the same 44 development questions produced
78 candidate requests. The maximum rendered prompt was 4,473 bytes, and the
maximum typed request total per query was 10,383 bytes. Both fit the existing
5,120/64,000-byte limits. Dataset and bound source hashes stayed unchanged.
This supersedes the preliminary prototype's size estimate for this candidate;
both measurements remain recorded. No provider call or quality score occurred.

Runner `640457055dde7a5f653fe54713a7a2bd8e6ff450` supports an explicit candidate
selection while retaining the old temporal treatment as its default. The
[prospective v5 protocol](../../plans/2026-11-06-answer-support-checked-v5.md)
freezes the paired study and its existing model, data and limits before
inference. At preparation, the comparison is NOT_STARTED.

The [receipt](answer-support-checked-v5-preparation.json) binds implementation,
tests, request construction and the isolated runtime wrapper. All 20 private
archive members verified; archive SHA-256:
`938e96b4fc291d4a8a4fbad2d3725d02ca10186c3e41899d040deb57a8df4684`.
The installed wheel and native-host receipts retain their earlier product
bindings. No new installed check, native pass, quality gain, fresh evaluation,
pilot day, live configuration change or publication is claimed here.
