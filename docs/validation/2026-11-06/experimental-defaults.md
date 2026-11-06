# Experimental defaults validation

All five module switches now default to off. Provider selection does not enable
them. Explicit configuration and environment overrides remain supported. The
embedded ingestion and consolidation paths now honor their module switches;
explicit remember remains available with no model calls. Disabled consolidation
preserves previously buffered evidence.

The setup form changes module flags only when a preset is selected. Ordinary
edits preserve configuration inheritance instead of copying global flags into a
project override. Starter SQLite selects all modules off. The research preset
selects sleep extraction and embedded recall gating; its description identifies
both effects and makes no quality claim.

The initial broad run produced 2,586 passes, 58 failures, 19 fixture errors and
4 skips. Failures exposed research tests relying on enabled defaults, absent
archives in the sparse checkout, and one expected response missing the tokenizer
metadata added earlier. Required archives were materialized byte-for-byte from
Git. The dataset's recorded SHA256 matched. No historical evidence was edited.

After explicit opt-ins and the expected metadata were added, all 264 tests in
the 26 affected files passed. Every one of the initial 77 failed or errored
cases was matched to a passing case in the resulting XML. Five installed-artifact
harness tests also passed. Black, strict mypy on the six affected source modules,
JavaScript syntax checking and `git diff --check` passed. The complete suite was
not rerun after those fixture updates.

The [check receipt](experimental-defaults-checks.json) records counts, files and
log hashes. Raw logs remain in a private ignored archive. These are code and
contract checks; they establish no retrieval-quality improvement, native-host
acceptance or pilot use. The prior installed receipt remains tied to product
`24887ef3`. A new artifact check is required for this later change.
