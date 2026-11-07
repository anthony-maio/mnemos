# Temporal support implementation checkpoint

Source `c320c3ef13c7a8f993fe0b6290f166cee00594f8` adds the experimental
`qualified_v3` answer-support option. It requires an explicit reference date,
checks temporal and conditional applicability, and supplies comparison evidence
from the same visible accepted candidates. Each selected answer must still cite
its own retained source evidence. The ordinary retrieval path and the default
support protocol remain unchanged.

Comparison evidence makes the whole candidate set a dependency. Correction,
forgetting, source changes, scope changes or a previously ineligible candidate
becoming eligible invalidate the dependent result. The checker revalidates this
set around provider work and before returning. It also copies provider state
between calls and counts repeated state against the aggregate request limit.
Both support protocols can receive the same reference date for comparison.

Validation passed 63 focused support, comparison and local-judge tests, plus
17 dataset-validation, judgment and memory-firewall tests. New cases cover the
comparison dependency, explicit date contract, state isolation, repeated request
bytes and dataset drift before provider setup. The initial ten failures are
retained alongside the final passing output. Black passed for six changed
Python files and strict mypy passed for three source modules. The earlier full
suite at `3446220c` is a separate checkpoint; it was not rerun for this change.

The offline prompt preflight used mocks and made no model calls. It covered
22 scenarios and 44 questions: 27 answerable and 17 without a supported answer.
The largest rendered prompts were 2,482 bytes for the baseline and 4,445 for
the new option, below the proposed 5,120-byte limit. These measurements do not
establish semantic accuracy.

The [prospective v4 protocol](../../plans/2026-11-06-answer-support-temporal-v4.md)
freezes both datasets, the model, runtime, thresholds and all 352 planned
observations before inference. The original twenty questions are regression
data; the additional twenty-four were designed after reviewing v3 and are also
development data. The comparison started from `c320c3ef` in an isolated local
runtime. Its eventual results belong in a separate report, including timeouts
and other failures. No fresh evaluation, extraction result, pilot day or release
is established by this checkpoint.

The [preparation receipt](answer-support-temporal-preparation.json) binds the
source, tests, protocol, datasets and retained validation output in a private
archive. Live memory stores and host configuration were not changed.
