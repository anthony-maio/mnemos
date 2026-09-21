# Exact-match precision labels

The summary exposed supported_precision and extraction_precision without the
proxy qualification present in the adapter notes and README. A downstream reader
could mistake exact-text matching for measured semantic factual precision.

Both aliases now carry exact_frozen_text_scope_and_evidence_proxy and an explicit
false semantic_factual_precision_measured flag. All-observation and complete-only
summaries receive the same labels. A separate semantic_factual_precision gate
remains NOT MEASURED. Numerical scores, intervals and thresholds are unchanged;
existing raw runs and historical artifacts are untouched.

The regression first failed on base 62b3bff with KeyError for the missing gate.
After the repair, 36 benchmark tests and 13 generation/scoring tests pass. The
regression checks both populations and both aliases, including preservation of
the original 1.0 fixture score. Black passes for the changed Python files.
Python 3.10 compatibility mypy passes for the changed harness using the existing
compat-py310 environment. The initial main-environment mypy attempt failed on
NumPy dependency stubs requiring Python 3.12, before checking this patch.

This changes benchmark reporting only. It supplies no semantic quality result,
model call, production adoption, fresh full-suite or installed-package claim.
The paired product checkpoint remains 995d9c2. Semantic evaluation and T18
acceptance remain open.
