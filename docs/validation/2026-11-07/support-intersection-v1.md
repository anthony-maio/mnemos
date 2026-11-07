# Intersection of saved support decisions

Requiring agreement between the saved QA and Qwen contrast selections improves
the recorded development result: 25 of 27 answerable questions have clean hits,
16 of 17 no-answer questions receive empty selections, and one unsupported fact
remains. This is a post-hoc intersection of existing outputs. No combined runtime
or fresh support evaluation ran.

## Matched comparison

Both components saw the same frozen hybrid candidates for 44 reused development
questions. The analysis verified exact question text, reference date, snapshot
identity, baseline order, fact content, version, scope, assertion status and
retained evidence before joining decisions. It kept a complete fact only when
both methods selected its ID, preserving baseline order. No answer label entered
the selection rule.

The QA component is the corrected v2 source-offset decoder. Its three recorded
repetitions selected identical IDs; this comparison uses repetition one. The
Qwen3:14b contrast component has one recorded observation per question. Reusing
that output cannot create three independent combined runs. The seven additional
QA questions have no counterpart in this comparison and were outside its declared
scope before the join.

| Saved selection method | Answer hits | Clean hits | Correct empty abstentions | Unsupported fact selections |
| --- | ---: | ---: | ---: | ---: |
| Corrected extractive QA | 25/27 | 23/27 | 16/17 | 4 |
| Qwen3:14b contrast | 27/27 | 27/27 | 12/17 | 5 |
| Intersection | 25/27 | 25/27 | 16/17 | 1 |

The intersection removed three unsupported selections from QA and none of its
expected selections. It retained 26 fact selections, 25 matching expected IDs.
That 25/26 fraction is expected-ID selection precision on these known cases,
not a fresh estimate of semantic factual precision or extraction quality.
Choosing these two components was informed by their earlier results.

Three questions still fail. The trial-period batch-size answer and the audit-log
reviewer's name are omitted because QA selected neither. For the unmarked
retention conflict, both components select the longer-retention fact. Their
agreement does not resolve the conflict or make that selection supported.

Descriptive 95% Wilson intervals are recorded for the 27 answerable and 17
no-answer questions. They do not correct post-hoc method selection or dependence
within question families. The sample and review requirements for fresh acceptance
remain unchanged. Original component scores and artifacts are preserved.

## What this changes

The methods have partly different errors, giving a concrete reason to test a
fixed composition instead of another broad prompt rewrite. A runtime composition
would still need to bind both decisions to one current candidate snapshot and
revalidate scope, versions and evidence after provider awaits. Provider failure
must remain unavailable, not an empty successful abstention. None of that is
established by an offline set intersection.

Latency remains a substantial limitation. Historical median component times on
these questions were about 223 ms for QA and 2,736 ms for contrast. These are
separate recorded calls, not a measured combined pipeline. Running only the
QA-selected candidates through Qwen would change its input and requires a new
experiment; this result cannot predict its decisions or speed. No claim is made
against the original 500 ms warm-recall requirement.

The analysis and independent count audit made zero model or retrieval calls.
All matched inputs and source hashes were checked, and the private archive was
verified member by member. The ongoing fresh retrieval-only evaluation kept its
original process and all bound files. Production source, defaults, host settings
and the inactive pilot are unchanged.

- [Analysis protocol](../../plans/2026-11-07-support-intersection-diagnostic-v1.md)
- [Machine-readable receipt](support-intersection-v1-receipt.json)
- [QA component results](extractive-answer-support-v1-v2.md)
- [Contrast component results](answer-support-contrast-v6.md)
