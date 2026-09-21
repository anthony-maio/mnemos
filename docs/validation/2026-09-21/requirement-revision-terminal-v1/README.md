# Requirement revision development v1

Ten authored raw-source development cases were committed and frozen before any model execution. Candidate 50200c0 uses local Qwen2.5:14b for selection and relationship-based reconciliation; preparation.json binds exact source, model and configuration. This corpus does not replace or weaken the original sixteen-case pipeline expectations.

All ten cases return runtime-valid results and match their frozen operation, role, kind, revision-intent and content requirements. Ten selection calls and eight reconciliation calls produce five exact-source proposals and three review outcomes. The counts include intentionally unresolved cases and discarded input, so this is not a semantic accuracy or release score.

Complete positive and negative replacement rules are added when no prior target is observed, and correct the expected version-3 target when one is observed. Duplicate old policies remain unresolved. A referential rejection with no approved replacement remains unresolved. An assistant report stays attributed conflict without correction authority. Questions and temporary formatting instructions do not reach reconciliation. The qualified replacement rule retains both the rule and its exception in one exact contiguous excerpt.

These are small authored development families, not the untouched acceptance corpus. The separate original pipeline run still exposes missed revision-intent metadata on its rejection-prefixed side-panel rule. That limitation is not erased by this corpus's matches.

The 129-test focused suite and Python 3.10 checks passed before the model run. The isolated package-initializer repair was made before this corpus was invoked; original and corrected preparations are preserved. No database writes or production adoption occurred.
