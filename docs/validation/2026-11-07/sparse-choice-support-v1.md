# Sparse choice support: faster output, failed quality

Reject this format for runtime use. Across 136 paired observations per arm,
the sparse format reduces median support time from 2,610.06 to 693.29 ms but
introduces 72 contract failures and retains only 6 of 80 answerable hits. Its
1,342.82 ms p95 also exceeds the unchanged 500 ms complete recall budget before
retrieval is included. Production support remains unchanged.

The independent audit passed. That verifies the run and its recorded failures;
it does not mean the candidate passed its quality or latency requirements.

## Paired development results

The comparison uses 44 known questions and 24 newly written development
questions. There are 68 unique questions, with 40 answerable and 28 absent.
Each arm ran twice on the same complete candidate cohorts and source excerpts.
Repeated observations do not create additional independent semantic cases.

| Suite | Arm | Repetition | Answerable hits | Clean hits | Correct empty abstentions | Unsupported selected facts | Operational failures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Known 44 | Baseline | 1 | 27/27 | 27/27 | 12/17 | 5 | 0/44 |
| Known 44 | Sparse | 1 | 2/27 | 2/27 | 2/17 | 0 | 26/44 |
| Known 44 | Baseline | 2 | 27/27 | 27/27 | 12/17 | 5 | 0/44 |
| Known 44 | Sparse | 2 | 2/27 | 2/27 | 2/17 | 2 | 25/44 |
| New 24 | Baseline | 1 | 12/13 | 12/13 | 9/11 | 2 | 0/24 |
| New 24 | Sparse | 1 | 1/13 | 1/13 | 6/11 | 0 | 11/24 |
| New 24 | Baseline | 2 | 12/13 | 12/13 | 9/11 | 2 | 0/24 |
| New 24 | Sparse | 2 | 1/13 | 1/13 | 6/11 | 0 | 10/24 |
| Combined | Baseline | Both | 78/80 | 78/80 | 42/56 | 14 | 0/136 |
| Combined | Sparse | Both | 6/80 | 6/80 | 16/56 | 2 | 72/136 |

Operational failures count as neither hits nor abstentions. The sparse arm's
smaller unsupported-selection count accompanies severe answer loss; it is not
an improvement in answer precision. Expected-ID precision among selected facts
is 84.78% for baseline and 75.00% for sparse. Neither reaches the 95% requirement.
Baseline misses the new historical ownership question in both repetitions and
continues to make unsupported selections. It is not an acceptance reference.

## Timing and output size

| Suite and repetition | Baseline median / p95 | Sparse median / p95 |
| --- | --- | --- |
| Known 44, repetition 1 | 2,617.15 / 2,768.78 ms | 711.17 / 1,342.82 ms |
| Known 44, repetition 2 | 2,609.64 / 2,700.43 ms | 696.69 / 1,326.97 ms |
| New 24, repetition 1 | 2,633.06 / 3,806.55 ms | 577.16 / 1,380.15 ms |
| New 24, repetition 2 | 2,597.92 / 3,850.72 ms | 531.00 / 1,330.97 ms |
| Combined | 2,610.06 / 3,806.55 ms | 693.29 / 1,342.82 ms |

| Combined measure | Baseline median / p95 | Sparse median / p95 |
| --- | --- | --- |
| Generated tokens | 143 / 212 | 33 / 63 |
| Provider generation | 2,346.60 / 3,488.32 ms | 525.87 / 1,055.29 ms |
| Provider prompt processing | 166.33 / 235.24 ms | 163.33 / 229.92 ms |

Sparse finishes sooner in all 136 pairs; the median paired change is
-1,959.22 ms. Source selections or assessed/unavailable status differ in 114
pairs. All timings include failed observations. Percentiles use nearest rank.
Provider component percentiles should not be added together.

The support timer covers snapshot construction, prompt construction, provider
execution, decoding and the normal support result. It excludes model loading,
warmups, fixture seeding and grading. No retrieval, extraction or QA model ran.
These are support component measurements, not complete recall or hook timings.

## Why the candidate failed

All 72 contract failures contain explicit zero probability entries. The frozen
format requires zero entries to be omitted. An offline diagnostic removes only
those entries from temporary copies and confirms that the remaining shape and
mass pass the original decoder. It does not reselect sources, regrade the run,
replace raw replies or change the recorded operational failures.

That rejection is only part of the problem. Of the 64 originally valid sparse
replies, 47 are answerable and only six hit. Another 17 have absent answers,
with 16 correct abstentions. In 54 valid replies every candidate receives the
first option, `different_request`. Relaxing zero omission would not repair these
valid semantic misses.

The experiment changes the output representation, compacts prompt JSON and
adds explicit index maps. It therefore measures that whole treatment. The
saved replies cannot establish which individual format change caused the
semantic regression, or whether a particular runtime schema constraint was
ignored. No new inference was used for this diagnosis.

## Execution and evidence

Source `42a6335159c60d17f0ce77b6221b5d40f0fe85f9` adds the experimental codec,
36 contract checks, the new development fixture and the prospective protocol.
The production files remain identical to product checkpoint `3d1866d5`.
The comparison uses local Qwen3:14b Q4_K_M under Ollama 0.40.9, temperature zero,
thinking disabled and 8,192 context tokens. The original 0.8 support threshold,
0.01 probability-mass tolerance, 5,120-byte prompt cap, 1,024 output-token cap
and 15-second operation deadline are retained. Maximum rendered prompt sizes
are 4,902 bytes for baseline and 4,960 for sparse; no prompt is truncated.

All 272 planned observations completed. There are 274 model requests, raw
responses and HTTP attempts, including one unrelated warmup per arm. No retry
or omitted observation occurred. Question order is frozen, first-arm ordering
alternates within each pair and reverses in repetition two. Labels remain
separate from model inputs.

The audit independently reconstructs distributions, selected sources, versions,
grades and paired results from the raw records. It verifies 174 bound files,
eight model/runtime assets, 68 disposable databases and their 127 fact and
evidence pairs. The database projection and file bytes are unchanged. These
stores reconstruct declared accepted facts and evidence; they do not reconstruct
original embeddings, events, jobs or lifecycle execution. Model unload, owned
server exit, closed private port and recorded process exit all verify.

The archive retains the initial setup failure: preparation incorrectly required
every old candidate to be eligible. The corrected preparation preserves the
original 78 eligible and four conflict-marked records, then verifies all 44 old
semantic requests exactly. That correction occurred before any model calls.
A separate auditor preflight shell-quoting failure and its corrected script are
also retained. The run's independent audit passed on its first execution.

The 90 focused checks pass, including all 36 new codec checks. Strict mypy and
Black pass. All 12 documentation consistency checks also pass. The prior
3,560-test full regression belongs to its original source;
it does not include these new experimental checks.

The private archive has 425 verified members, 1,692,134 bytes and SHA-256
`5f755a000c2ae40704db7c5760856d38977dac3fab580f72ca7ca70ac1f92c78`.
The receipt includes all paired differences and the failure diagnostic counts.

## Next work

Keep this candidate disabled. Any follow-up should retain named option identity
and complete probability mass, separating output serialization from prompt
changes. Freeze a new comparison before inference and retain this failed run.
A smaller output alone will not justify adoption: supported-answer retention,
abstention and the complete latency budget must all pass on fresh acceptance
data after development.

The existing support, extraction and retrieval results remain unchanged. Current
quality gates, complete hook latency, canonical delivery and the actual private
daily-use pilot remain open. The pilot is inactive with zero days; no host
configuration, production default or publication state changed.

- [Frozen protocol](../../plans/2026-11-07-sparse-choice-support-v1.md)
- [Aggregate receipt](sparse-choice-support-v1-receipt.json)
- [Earlier live support composition](support-intersection-live-v1.md)
