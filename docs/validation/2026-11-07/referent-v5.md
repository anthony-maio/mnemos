# Reconciliation referent screen v5

The value-span treatment improved the decision score to 19/20, with all nine
review controls correct, no accepted wrong targets or additions, and no
operational failures. It still missed the original single-target correction
and fails the frozen confirmation gate. Keep it experimental.

The [protocol](../../plans/2026-11-07-reconciliation-referent-v5.md) and source
were frozen at `90ae6cfdb5fba233871f9987b18781df3780bfe9`. Twenty supplied
classifications and target snapshots ran once under three treatments. The two
identity treatments share routing and operation rules. The new treatment
selects literal value spans by word ID, then judges whether the statements
identify the same variable, different variables or insufficient identity.
These are known development cases from six clusters, not fresh acceptance.

| Measure | Direct/full | Quoted identity | Value-span referent |
| --- | ---: | ---: | ---: |
| Exact operation and target outcomes | 17/20 | 15/20 | 19/20 |
| Correct unresolved controls | 6/9 | 7/9 | 9/9 |
| Accepted wrong support or correction targets | 3 | 0 | 0 |
| Accepted unsupported additions | 0 | 1 | 0 |
| Operational failures | 0 | 2 | 0 |
| Median observation latency | 1,410 ms | 2,481 ms | 1,795 ms |
| p95 latency, nearest rank | 1,799 ms | 5,393 ms | 3,200 ms |
| Logical calls and recorded responses | 20 | 17 | 17 |

Compared with quoted identity, the new treatment recovered the qualified
correction, the target among distractors and the requirement reversal. Both
quote failures became valid review outcomes. It lost the simple known-target
correction: the candidate and target named the same backup window, but the
model labeled them different variables. The same candidate and matching target
text received a match in the distractor case. Target ID and list context also
differ between those cases, so this is a diagnostic inconsistency rather than
an isolated test of distractor effects.

The decision score also hides intermediate errors. A post-run review of the
17 new comparisons found seven relations inconsistent with the stated contract:
the known correction and its two duplicate-target copies were called different
variables, and four comparisons with missing numeric or environment identity
were called different rather than insufficient. Six of these errors still
led to the expected review operation. This diagnostic was annotated after the
run; it is not a prospectively frozen relation-quality score and does not
replace any original operation score. It shows why correct final review alone
does not establish a reliable identity comparator.

All 27 selected value spans were ordered and within their original source
inventories; two value fields were null. Some selections covered whole clauses
or included property words, so valid offsets do not establish accurate value
extraction. The missing-target and exact-duplicate routes still require no
model call. Non-user history retains its separate relationship path and cannot
gain revision authority through this comparator.

| Provider accounting | Direct/full | Quoted identity | Value-span referent |
| --- | ---: | ---: | ---: |
| Prompt tokens | 14,747 | 10,250 | 17,520 |
| Cached prompt tokens | 11,137 | 8,268 | 9,503 |
| Generated tokens | 1,127 | 2,671 | 1,407 |
| Final text characters | 5,099 | 10,817 | 5,305 |

All 54 requests returned one response with a normal stop reason and no thinking
text. The two quoted-identity failures came from runtime quote validation,
not missing responses. Every treatment used Qwen3:14b Q4_K_M, Ollama 0.40.9,
generate mode, thinking false, temperature zero, context 8192 and one request
slot. The shared limits were 4096 output tokens, 3584 counted input tokens,
a 512-token template reserve, a 90-second provider deadline and a 120-second
observation deadline. Maximum native prompt-plus-schema, reported prompt and
generated counts were 1,942, 1,442 and 299. No output limit was exhausted.

Preflight verified all 63 expected alternatives across the three arms and
replayed the previous 60 identity observations with identical prompts,
decisions and failures. The independent audit rebuilt all 60 new scores,
replayed every result or failure without model calls, checked 136 source
hashes and all 20 identical input groups, and inspected the 17 comparisons
and 29 value fields. The 159 focused tests, including 22 new checks, and 12
documentation checks passed. Black and strict mypy passed for the changed
source. The earlier full-suite result keeps its original source binding.

The [receipt](referent-v5-receipt.json) binds 176 verified private archive
members under SHA-256
`07fecb110d705eba2f6d61818ab0ebbe0632e9c29cf37e96026a15e8392ea06c`.
The archive includes the post-run semantic diagnosis and the previous identity
records used for replay. The model unloaded, all listed owned processes were
absent and both endpoints closed. The original tree-stop warning remains
preserved, with no extra kill or inference rerun.

The narrower contract improves this development decision score but does not
meet its selection rule. Further work must evaluate the intermediate relations
and their stability as well as final operations. Production defaults, live
configuration, installed pilot artifacts and publication state are unchanged.
Full extraction, persisted outcomes, fresh quality and actual daily use remain
open.
