# Qwen2.5 14B extraction review

All 92 calls were runtime-valid. Neither prompt is adopted. Both lose the
required marker in mixed-marker-13-policy and infer a possible user preference
from an unaccepted assistant suggestion in N03. Both retain unwanted speculation
in N02. Candidate N08 proposes support for an unrelated existing fact with a
source that does not support it. No facts were committed in this diagnostic.

The 14B outputs preserve P05 open consideration, which the earlier 9B outputs
lost. That improvement does not erase the negative controls or omissions.
The model and thinking configuration both differ from 9B; this is a bundled
configuration comparison, not causal evidence about model size or thinking.

review.json contains all 46 source/rubric/proposal comparisons. Original
ambiguities remain explicit. This is one unblinded stochastic repetition of
reused development cases, not held-out accuracy, native recovery or module
benefit. Raw outputs and configuration are preserved in the sibling
extraction-model-comparison-14b-terminal-v1 archive. The current service write
firewall is not exercised by proposal-only success; evidence entailment is a
separate quality requirement from ID validity.
