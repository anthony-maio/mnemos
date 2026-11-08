# Attribution of the closed support comparison's failures

A read-only audit recomputed all 246 saved QA decisions and checked all 272
original observations against their recorded grades. It reproduced every
intersection selection from the two component selections, verified literal
span matching against the returned facts, and confirmed that no final budget
omission explains a lost answer. No inference, retrieval rerun or score change
was performed.

The eight lost answers are four questions repeated twice. The four unsupported
selections are two questions repeated twice. Repetitions are not independent
examples.

| Failure | Observations | Component behavior |
| --- | ---: | --- |
| Reviewer identity | 2 | Contrast accepts; QA prefers its no-answer score |
| Value during an explicitly dated trial | 4 | Contrast accepts; QA prefers its no-answer score |
| Historical owner before replacement | 2 | Both components reject |
| Current value with unresolved conflicting statements | 4 | Both components accept the same unsupported selection |

The first-repetition QA span-minus-null margins for the four lost-answer
questions are -3.11, -21.55, -15.71 and -27.14. The two unsupported selections
have positive margins of 20.41 and 21.10. These raw logit differences are not
probabilities. The correct answer text appears as the best source span in each
lost-answer case, but the decoder rejects it in favor of the null score.
Literal source presence alone also fails to resolve the conflicting statements.

Eight candidate exclusions across four observations carry an explicit stored
conflict status. The audit verifies their `stored_conflict` decision and absence
from both component selections. Those exclusions are intentional and separate
from the unsupported selections where conflicting statements remain asserted.
The first audit attempt incorrectly expected QA output for these excluded
candidates and stopped at that assertion. Its script is retained. The corrected
audit verifies the explicit exclusion instead of inventing missing model output.

## Consequence for the next design

The existing treatment stays closed and outside defaults. The evidence supports
investigating a different inference design with two capabilities: answering
historical questions within an explicit validity interval, and detecting
incompatible answers across the candidate cohort without suppressing unrelated
facts such as the reviewer identity. A per-source answer span cannot establish
the second property. Agreement between these two models did not establish it
either.

The next candidate should evaluate a bounded, batched discriminative support
path rather than another generative output codec. This is a design direction,
not evidence that a particular model works or runs within the latency budget.
Before selecting a model or training approach, inspect its supported inputs,
license and inference requirements. The development contract must distinguish
question answerability from pairwise evidence compatibility, retain temporal
qualifiers, preserve source/version fences, and measure complete retrieval plus
support latency. Batching alone does not guarantee the 500 ms requirement.

Do not fit thresholds to these six failures or convert their original grades
into successes. Use the known development corpus to diagnose a separately
frozen candidate; reserve fresh acceptance data until that candidate passes the
original development requirements. Critical-fact labels, host-token costs and
10,000-fact latency remain separate unfulfilled obligations.

## Evidence

The private audit is `.venv/support-failure-attribution-v1/audit.json`, SHA-256
`d3883d0b5453d89c9988bdf172e8ea1d9aca429e520294be2a65adf610e4d7c2`.
It includes all 12 failure observations, their original queries, source spans,
component attribution, and SHA-256 bindings for the six input files and decoder.
All seven bindings match before and after execution.

The [original comparison](retrieval-support-development-v1.md), its frozen
archive, failed quality and latency gates, and all per-question grades remain
unchanged. No product source, live configuration, memory database, pilot status
or publication state changed.
