# Evaluation registry identity repair: focused verification

Source 1b3e227 passes 105 selected benchmark/contract/launcher checks, Black,
and the Python 3.10 compatibility type check. The regression file covers eleven
cases, including conflicting event/query/state-pair contexts, inherited project
identity, workspace profile conflicts, consistent sharing and rejection before
provider discovery/output creation. The initial six-case run failed five cases
before the repair; its results motivated the fix, not a threshold change.

This is focused evaluator verification, not a new full product/package checkpoint
or a semantic-quality result. Frozen v2-v4 data and historical scores remain
unchanged. The README correction states the existing exact-text proxy behavior:
unmatched paraphrases count false rather than unjudged. No scoring code changed.

The first direct mypy invocation used the main Python environment and failed
while parsing installed NumPy stubs under the configured Python 3.10 target.
The recorded passing command uses the established compatibility environment,
as do the source/package checkpoints; no checking rule was disabled.
