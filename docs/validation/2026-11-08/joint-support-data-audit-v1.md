# Grouped support-data integrity audit

Commit `1fd3a2ba` adds the inactive `joint_support_data` audit. Corpus examples carry inference requests separately from split/lineage metadata. Labels are separate records bound to each canonical request hash and exact target IDs. This keeps gold labels out of the serializer and detects stale labels after input changes.

The audit joins shared scenario, source, template, paraphrase and counterfactual families transitively. Normalized fact/source content adds conservative overlap edges even when ancestry is renamed. Any connected group crossing train, tuning or development splits is rejected. Expanded variants count as examples but not independent families. Normalization covers case, Unicode compatibility and whitespace; it cannot discover arbitrary undeclared paraphrases or prove that all lineage declarations are truthful.

Duplicate/orphan/missing targets, inconsistent supported/reason labels, contradictory labels for identical target inputs, unknown partitions and known material in reserved tuning/development partitions are rejected. Acceptance is not a valid partition for this development-data API. It reports positive/negative reason coverage, supported critical targets and pending/reviewed/disputed annotations. Structural success never marks labels semantically verified or training ready. Positive and negative rows in each partition are only a basic coverage check, not a sufficient diversity or quality criterion.

The combined serializer/audit suite passed 35 tests in 0.82 seconds, including transitive overlap, renamed normalized evidence, stale bindings, same-split augmentation, contradictory gold, and rejection of acceptance rows. The earlier 34-test run and final run are retained at `.venv/joint-data-v1.xml` and `.venv/joint-data-v2.xml`. Formatting and `git diff --check` passed.

This validates audit mechanics on fixtures, not an actual training corpus. The reviewable corpus, independent label audit, padding/shared head, computational preflight and training recipe remain open. No model call, training, semantic evaluation, production change or pilot activity occurred.
