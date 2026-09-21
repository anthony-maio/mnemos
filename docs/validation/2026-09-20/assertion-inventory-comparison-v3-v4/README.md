# Qualified inventory configuration comparison

All 46 original cases ran once in each configuration. V3 produced 31 valid
inventories, 13 empty-final budget failures and two source-role rejections.
V4 produced 40 valid inventories and six validation rejections: three source
quotation failures and three source-role failures. No v4 empty-final failures
were observed. These counts describe execution, not semantic accuracy.

The thinking-disabled configuration also selects generate instead of chat.
The comparison cannot isolate thinking or establish a general speed advantage.
Both use the same frozen model digest, inputs, candidate, output budget and
original rubrics. All raw outputs and descriptive reviews remain available in
the adjacent terminal and semantic-review directories. Neither is adopted.

The paired review identifies three distinct repair needs. Retention intent is
not established by finding an exact quotation: N02 still mistakes thinking out
loud for a request to remember. Source validity does not constrain paraphrased
meaning: P06 changes possible future acceptability into permission, while P01
includes classification commentary in memory content. Finally, a condition can
be omitted or stored separately, leaving the retrieved assertion incomplete.

Next candidate work must address those three boundaries explicitly, retaining
mixed-message facts, corrections, qualified history and requested uncertainty.
A blanket refusal to remember ambiguous messages would fail the original goal.
Keep evidence selection and memory wording separately inspectable, and ensure
conditions needed to interpret a memory travel with that memory. Validate the
complete original development set before any held-out run or production change.
Do not relax exact quotation checks, accept rejected outputs, or infer semantic
success from runtime recovery. Reconciliation and full-module acceptance remain
separate open gates after source-only extraction.
