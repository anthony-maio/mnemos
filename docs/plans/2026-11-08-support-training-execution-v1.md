# Training execution contract

Status: bundle and tokenized frames prepared; the training/evaluation worker is
not implemented or execution-frozen yet. This document fixes comparison and
selection rules before any trained outcome is observed. It does not change the
original quality, latency, native, publication or pilot obligations.

Use corpus-v3 with its recorded inputs/labels/critical-registry hashes. Training
targets and weights contain only the 67 training queries and all 282 targets.
Reserved decisions cannot change the bundle. Examples passed to tokenization
contain only request fields, never labels, reasons or critical flags. Original
label files remain separate for diagnostic grading after predictions are saved.

The bundle fixes 50 optimizer updates, full-partition gradient accumulation,
rank-8 q/v adaptation, alpha 16, zero dropout, seed 2026110808, AdamW groups,
gradient scaling and clipping from the reviewed learning recipe. Each update
uses every training query once. The final completed update is the only selected
checkpoint. No threshold, epoch, seed, sample weighting or attention sweep is
permitted. A 1800-second outer worker timeout includes loading and diagnostics;
timeouts, nonfinite values or skipped updates are failures with stage evidence.
The execution manifest must bind the actual worker, helpers, plan, full runtime
distribution versions, device/driver, model files, bundle and tokenized frames.

## Matched-evidence baseline

Use the original 28-layer reranker checkpoint, original instruction, chat prefix
and suffix, no/yes token IDs and decoder threshold >=0.9. Each owned source
produces one witness containing target identity/content/evidence and the full
ordered candidate cohort. Retain query, date, candidate version/status and exact
source owner/event/excerpt. Omit absent timestamps/source refs; do not manufacture
evidence from fact text, merge witnesses or globally deduplicate source IDs.
Any passing witness selects its fact, as in the original decoder.

Both systems exclude targets without owned sources or whose status is not
asserted; excluded peers remain in context. All corpus-v3 targets are asserted.
The one source-less development target gets no baseline witness and cannot be
selected by either decoder. That alone does not make the entire question an
operational failure. An unavailable question stays in the grading denominator
and earns neither correct abstention nor exact-cohort credit. More than 32
witnesses, any baseline frame over 4096 tokens, nonfinite output or malformed
identity invalidates the complete question, without truncation or partial output.

Asserted-only eligibility is a disclosed shared comparison guard, slightly
narrower than the old baseline's explicit-conflict exclusion. It changes none
of the current 94 development targets. Do not describe this as a general proof
of identical eligibility outside the frozen corpus.

Development has 48 questions, 94 targets and 109 owned-source witnesses. The
prepared baseline frames span 338 to 720 tokens; the joint frames for all 127
cases span 105 to 552. These are matched evidence, not identical token/model
inputs. Different witness versus joint policies, topology and training prevent
attributing any gain solely to one factor. Old baseline scores remain unchanged.

## Prediction and artifact decisions

Retain an initial untrained prefix/head control, the unchanged reranker baseline,
the final trained unmerged output and the final merged output. No intermediate
checkpoint competes for selection. Joint selection uses finite logit >=0 and
the stated deterministic eligibility rule. Save raw target IDs/logits and
witness scores; reconstruct selections from the frozen decoders when grading.
Tokenized frames must match their saved bindings before every phase.

Before accepting a merged artifact, compare merged and unmerged full-model
float16 inference on every corpus request and all six capacity workloads.
Maximum absolute logit error must be <=0.02, with exact logit>=0 agreement for
every target, including ineligible targets, and identical final selected IDs.
Nonfinite output fails. Do not relax tolerance or hide a threshold flip behind
eligibility. A failure preserves both outputs and makes the merged artifact
ineligible; it does not authorize another checkpoint or threshold. The float32
linear unit test is not this full-model parity gate.

Grade tuning and development separately after final predictions are recorded.
Report supported precision, supported-target and critical retention, query-level
answerable recall, no-answer abstention, whole-cohort accuracy, category errors,
availability and permutation consistency. Report distinct critical facts with
their individual opportunities. Diagnostic thresholds remain >=95% precision,
supported retention and critical retention, >=90% recall and abstention, with
all observations available. Perfect diagnostics on three critical development
facts cannot satisfy original fresh acceptance or independent repetition minima.
Neither this training run nor its support-only timings prove complete retrieval
latency at 10000 accepted facts. No installed configuration changes follow.

The grader identifies its split and names each critical fact's applicable,
retained, missing and unavailable cases. Permutation consistency is reported
separately rather than added as a new threshold; diagnostic_thresholds_met
does not mean every contrast or permutation passed.
