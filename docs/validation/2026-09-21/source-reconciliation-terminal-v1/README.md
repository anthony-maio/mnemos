# Reconciliation development result

Local Ollama qwen2.5:14b, temperature 0, think false, 8192 output tokens, 180-second timeout. Exact digest and provider source hashes are in preparation.json. Candidate commit: fb88796. Nine provider-invoked cases returned mechanically valid decisions. The tenth rejected a foreign target before calling the provider. No memory writes occurred.

Five of ten cases match the frozen expectations, including the deliberate missing-target outcome, standalone-rejection contract gap, and foreign-scope rejection. This count is not a memory accuracy score.

Failures: independent new facts became unresolved; exact existing facts became add rather than support; the multi-candidate case made both errors; duplicate correction targets produced conflict rather than the required unresolved decision. The duplicate guard still prevented choosing either identity. The model's reasons invented candidate versions or claimed an observed target did not exist. These explanations are unsupported by the inputs.

The known-target correction selected the expected identity and version. Assistant-attributed disagreement remained a conflict without changing the user fact. Standalone rejection remains unrepresentable as a new retained constraint under the current contract.

Next work: clarify that supplied targets already exist, additions need no target, and candidate source positions are not fact versions; enforce exact normalized support through deterministic identity handling where unambiguous. Preserve this run and its original criteria. Do not promote this candidate to production or claim end-to-end acceptance.
