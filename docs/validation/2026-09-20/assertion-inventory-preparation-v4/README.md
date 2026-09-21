# Source-only inventory thinking-disabled comparison preparation

This freezes the same 46 reused development cases and qualified-inventory v3
candidate as the thinking-enabled Qwen3.5:9b experiment. Exact model digest,
provider source, prompt, schema, case order, temperature 1.0, 8192 output tokens
and 180-second timeout remain unchanged. The only generation-input difference
is think=false. In the frozen managed provider this also changes chat to generate,
so this is a configuration/route comparison, not isolated reasoning causality.
The candidate contract remains v3; v4 identifies this experiment preparation.

All 46 offline paired requests have identical prompts and schemas and valid
source bindings. Missing/wrong external preparation digests fail. A correct
digest also fails before generation while the parent experiment remains
unfinished. The launcher requires all 46 parent cases completed and matching
terminal observation bytes, then records that binding before starting.
No inference was performed during this preparation or its preflight.

The parent experiment remains unchanged. Review every original required and
forbidden meaning, preserving ambiguity notes, runtime errors and unwanted
retention. One stochastic repetition on reused unblinded cases cannot establish
held-out accuracy, full-module benefit or general reliability. Existing targets
remain absent; there is no reconciliation or durable write. No production
default changes or adoption follow from parse success. Hardware/Ollama inventory
is inherited from the bound parent preparation; model digest was checked again.

Status: NOT STARTED. Start only after the unchanged parent run is terminal,
using this archive's preparation digest. Do not retry only negative cases.
