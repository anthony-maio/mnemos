# Fixed eight-layer training attempt: failed

The single frozen attempt completed all 50 full-partition updates and failed
both semantic development and merged-model agreement. Close this candidate.
Do not select a different threshold, checkpoint, epoch count or depth, deploy
the unmerged model, or consume fresh acceptance cases to rescue this result.

Reviewed source `1002fd1b` ran once under the 1800-second supervisor and finished
in 212.83 seconds. Ten objective/worker tests and twelve audit/grading tests
passed before execution. The scoped implementation review found no blocking
objective, gradient, selection or merge-gate defect. Its limitation was explicit:
diagnostic grading alone does not verify saved artifacts or complete merge parity.

Weighted training loss decreased from 0.763396 to 0.262712. Every update recorded
finite loss/gradients, nonzero adapter-B and classifier gradients, and gradient
scale 128 with no skipped update. The original frozen parameters and all file
bindings remained unchanged. This optimization progress did not produce reliable
support decisions, even on training data: final unmerged training precision was
50.82%, supported retention 77.50% and no-answer abstention 48.28%.

Post-run review found no concrete source, binding or weighted-objective bug
that justifies rerunning the attempt. Recomputing weighted BCE from the saved
post-update training logits gives 0.258497, consistent with the final logged
pre-update loss. Semantic failure precedes merging: even the authorized quorum
case has unmerged logit -4.46094. The evidence supports insufficient learned
selection and order sensitivity, not an isolated cause in depth, data or loss.

## Development results

All 48 development questions were assessed, with no operational unavailability.
There were 40 supported target opportunities, 36 answerable questions and twelve
no-answer questions. Three critical facts supplied eleven supported opportunities.

| System | Precision | Supported retention | Critical retention | Answerable recall | Abstention | Exact cohorts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged reranker baseline | 43.01% | 100% | 100% | 100% | 0% | 29.17% |
| Initial prefix/head control | 43.01% | 100% | 100% | 100% | 0% | 29.17% |
| Final unmerged candidate | 34.78% | 20% | 0% | 22.22% | 58.33% | 25% |
| Final merged computation, ineligible | 37.50% | 22.50% | 0% | 25% | 58.33% | 25% |

The baseline and initial control selected every eligible development target;
matching selections do not mean their logits or inference paths were identical.
The trained unmerged model selected eight supported and fifteen unsupported
targets, missed 32 supported opportunities and correctly abstained on seven of
twelve no-answer questions. Its false supports comprised five wrong-relation,
four incomplete-source, four temporal and two conflict cases. It missed every
applicable occurrence of build-no-both, pass-rule and vault-quorum-veto-rule.
Three reordered-cohort groups also changed selections.

Tuning failed separately: unmerged precision 50%, supported retention 28.57%,
critical retention 0%, recall 28.57% and abstention 80%. No tuning outcome changed
the frozen checkpoint or decision rule. These are known grouped diagnostics,
not fresh acceptance or proof of the original critical-retention population gate.

## Merge and artifact verification

The independently recomputed merge check covers all 127 corpus cases and six
capacity workloads. Fifty-nine cases violated the frozen agreement requirement.
Maximum absolute logit difference was 0.0537109375, above 0.02. Two targets crossed
the fixed zero-logit boundary: box-rule in the exception-precedence scope case,
and unit-y-false-grade in the corrected source-ID-collision claim. Both changed
final selections. The slight merged metric differences do not make that path
eligible. No final-merged.safetensors was saved.

Independent checks verified all 50 update receipts, 133 unique assessed rows in
each joint phase, 48 unique assessed baseline rows, unchanged frozen bindings,
the complete raw-output merge calculation and its recorded failure list. The
saved final adapters contain exactly the expected 34 float32 tensors and 328705
parameters; their SHA256 matches the worker receipt. They are retained only as
failed experimental evidence, not an installed or approved artifact.

## Evidence and remaining work

Private run: `.venv/joint-support-training-v1`.
Freeze SHA256: `2956f081427b20a0c586023e096d1635db16dc224db164228fe3660ea5508cf1`.
Result SHA256: `1a0b97841378255d7e089005e2e4ea3fdd162e4f42f3055b2b5c3a3c674bdaf0`.
Diagnostic grades SHA256: `9dfb8582e3b3482b63dc25d9fea7b32d9e61783eea1bc6bba89de4c5939aa98a`.
Final adapters SHA256: `ec5f80f3b3b5d7f11eea538ea946464fea56a256532328de4a6e9b0301af17dc`.

The readback-verified `.venv/joint-support-training-evidence-v1.zip` contains
45 files plus a manifest, including frozen source/input snapshots, raw outputs,
training receipts, adapters and independent audit. SHA256:
`506b8c7585fc8539fc390feeddd3cb9861289930894970b72abcac15d8a94e1b`.
Retained base-model files are referenced by their frozen hashes rather than
duplicated in the archive.

Capacity success and a working backward path did not establish semantic success.
Any future semantic candidate needs a substantive reviewed design decision;
this result does not authorize a parameter sweep or another formatting variant.
Current artifact/native acceptance, complete latency, affected-user delivery and
the actual pilot also remain open. No live memory, host configuration or package
installation changed during this attempt.

Next substantive work is the affected-user storage delivery in draft PR #9:
identify and complete its remaining plugin/Desktop/pilot and final-artifact
release requirements, without repeating unchanged package checks or waiving
gates. Semantic quality remains an unmet part of the original goal. No new
model execution is justified by this failed candidate's saved outputs.
