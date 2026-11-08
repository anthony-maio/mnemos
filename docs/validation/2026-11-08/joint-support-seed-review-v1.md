# Training-only support seed and label review

Commit `09a2048e` adds a reviewable synthetic seed builder. The corrected v4 artifact contains 51 questions and 255 target labels across six scenario families. Shared constructions connect those scenarios into only two template groups. All examples are training-only. There are no tuning, development or acceptance examples, and all review statuses remain pending for final corpus adjudication.

The initial v1 draft assigned analogous domain constructions to different partitions and declared their templates independent. Its mechanical audit passed those declarations, but inspection showed that the independence claim was unjustified. V2 correctly joins the shared templates and moves every example to training. This is evidence of the limits of metadata checks, not a passed leakage review of v1. V1 remains preserved and is excluded from use.

Review then identified three incorrect negative labels: `larch-operations-8/c`, `museum-access-5/c` and `harbor-berths-5/c`. The rules plus explicitly unmet prerequisites establish a supported no answer. Answer support must not be confused with affirmative permission. V3 corrects these labels and rewrites three quotation meta-questions to ask about an actual action with genuinely unknown circumstances. Slot-based critical flags were removed; they did not define critical facts defensibly.

V4 additionally distinguishes retention policy from observed log retention, assigned berths from actual ferry movements, and visit dates from booking dates. Both target text and questions were corrected where needed. Earlier labels and bindings are retained. The corrected seed has 27 supported target occurrences, 198 wrong-subject/relation negatives and 30 other negatives. No target is designated critical. These counts are not independent critical-fact or quality measurements.

The structural audit confirms separate request-bound labels, two connected training groups and no reserved partitions. It correctly reports that minimum partition checks fail and training is not ready. The corpus is heavily patterned: every example has five candidates, one source per candidate and at most one positive target. Missing coverage includes multiple supported targets, source-ownership traps, multiple sources, compatible alternatives, conflicts without explicit disagreement hints, supported attribution, negated prerequisites, temporal boundary variation, source-embedded instructions and permutations. Generic reason strings do not replace case-specific label adjudication.

The next corpus increment must add substantively different constructions and appropriate positive/negative counterparts, not expand row count by swapping names. No training or capacity/model inference has run. This seed cannot be used as a new development or acceptance set merely by renaming a file.

Private final artifacts are under `.venv/joint-support-seed-v4`:

| File | SHA256 |
| --- | --- |
| inputs.json | 25e75b85b09c5fa488083bfe5c39aaf7d943e4e358e813e4e403cd3eb098eedd |
| labels.json | 70e5995bd30d8eb14801e1076949af69b9fdabef8bbfb905cbe026e0e53feeef |
| structural-audit.json | 8f683d3b1f7a78623c85da48c4d121631944d85566d1cd360fb95221290bf432 |

The 18-member `.venv/joint-support-seed-review-v1.zip` retains all four preparation states, v2's initial and corrected manifest prose, and the final builder. SHA256 is `15460118e6cf0175c5f628d667d63ccd1eda421cfb70574933b161e88e970d93`. Its contents were read back and matched against all retained files. These are synthetic preparation artifacts; no private project data, live memory, pilot setting or historical benchmark score changed.
