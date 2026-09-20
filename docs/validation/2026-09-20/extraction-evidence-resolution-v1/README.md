# Extraction evidence resolution repair

Source e082790 passes 113 focused and adjacent evaluation tests, Black, and
Python 3.10 compatibility mypy. The initial four-case reproduction had three
failures and one passing control: a valid source masked a missing event, an
unmapped dataset source, or an event without original-source metadata.

The evaluator now requires every declared source to resolve before accepting
the exact-text support proxy. Raw judgments expose source_resolution_complete.
This changes future scoring of malformed evidence; it does not establish
semantic accuracy or alter original criteria. Historical runs retain their
original scores and code bindings. No production engine, model prompt or
provider configuration changed.

These are focused checks, not a new full-suite or installed-wheel claim. The
latest full-suite evidence remains bound to 6ec8259. The ongoing durability
comparison uses its separately frozen source and is unaffected.
