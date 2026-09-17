# Validation environment change

At approximately 06:47 UTC on September 17, a validation command used `uv run`
and synchronized the shared Python environment. Its output reported two
uninstalled and two installed distributions. Metadata timestamps identify
`filelock 4.0.0` and the editable `mnemos-memory 0.7.0` installation. The prior
filelock version was not retained in that command's output. The generated lock
file is preserved privately at `.venv/environment-sync-generated-uv.lock`.

This overlapped the b7 held-out mechanism evaluation and the first 20f full
verification attempt. Neither frozen source checkout was modified. The audit
recomputed the b7 source hash and compared every runtime version recorded in
its manifest; both matched. A fresh process imported Mnemos and the evaluator
from the frozen b7 checkout, without importing filelock. A source search found
no direct filelock reference in Mnemos, the evaluation code, or the tests.
Those checks do not inspect the already-running process's module state.

The real-model run continues with this environment incident disclosed. Its
recorded runtime and source binding still match; no causal effect on its
observations has been established. Do not describe the broader environment
as unchanged. The current full test attempt is retained with this caveat and
should be followed by verification without an intervening environment sync.
No model result or threshold is discarded or adjusted because of this event.

The native Codex v6 result predates the sync and used a separate installed
wheel environment. Its evidence is unaffected by this shared-environment
change. Future validation commands use explicit existing interpreters, and
package installation uses a separate target or virtual environment.
