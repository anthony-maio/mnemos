# Training inputs and diagnostic grading preparation

The bundle, baseline adapter, tokenization preflight and diagnostic grader are
implemented. Thirty-nine baseline, grading, bundle, data-integrity and loss
checks pass. No training/evaluation worker has run; its implementation and exact
execution freeze remain next. No model weights were loaded for this preparation.

The bundle contains all 67 training queries and 282 weighted training targets.
Reserved labels are absent. Mutating reserved decisions leaves the bundle
unchanged, and reordering label records cannot reassign targets or weights.
It binds the exact corpus-v3 and original reranker instruction/frame artifacts.
The prospective recipe fixes 50 updates, final-checkpoint-only selection, joint
logit >=0, and full-model merge absolute logit error <=0.02 with exact threshold
decisions. These are declared requirements, not executed results.

The baseline preserves one witness per actual owned source and the original
any-witness >=0.9 decision rule. It retains complete ordered peer context,
source ownership, query, reference date and candidate version. Missing evidence
is not replaced with fact text. Asserted-only selection is a shared comparison
guard, narrower than the old baseline's explicit-conflict exclusion but with
no effect on the current asserted-only development corpus. This is matched
evidence with adapted serialization, not identical model input or a depth-only
ablation. Review found no leakage or evidence-mapping defect.

Retained-tokenizer preflight produced 127 joint frames, maximum 552 tokens, and
109 witness frames across all 48 development questions, lengths 338 to 720.
No overflow or truncation occurred. This preserves the declared 4096-token
bound; short corpus inputs do not redefine it.

The grader preserves question/target denominators, rejects missing observations,
and gives unavailable questions no abstention credit. All-abstention output
fails retention. Review identified missing per-fact reporting and split identity;
both are now implemented and tested. Critical facts list applicable, retained,
missing and unavailable case IDs separately from aggregate opportunity metrics.
Permutation consistency is reported, not an additional gate. A diagnostic
threshold pass cannot be described as every contrast passing or fresh acceptance.

Private bundle SHA256:
`45f85f64f194b0e45913392007e3a7c2c024fd40802079953ff555b318051872`.
Tokenization preflight SHA256:
`803bce271e0f8478f86cf7e8b0561be1ca47c83fb207275263c2c7ea5148278c`.
The readback-verified `.venv/joint-support-training-contract-evidence-v1.zip`
contains 15 files plus a manifest, SHA256
`01138ececb6d857f0d4220de081fea9a1f73749a9848406a68534d5bb2766855`.

The [execution contract](../../plans/2026-11-08-support-training-execution-v1.md)
now specifies the worker's outstanding implementation: unchanged baseline,
initial control, full-partition accumulation, fixed final checkpoint, merged
agreement, terminal receipts and preserved raw outputs. Training remains
unexecuted; semantic, complete-path latency, delivery and pilot gates stay open.
