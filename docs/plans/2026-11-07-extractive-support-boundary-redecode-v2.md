# Source-boundary decoder correction

The original extractive screen is complete and preserved. Its audit exposed a
tokenizer boundary mismatch: DeBERTa can represent the first source word with an
offset that includes the newline immediately before the excerpt. Strictly
requiring the entire token to lie inside the source then discards a valid answer.
The original run, raw logits, decoder, protocol and bootstrap failure remain in
the private v1 archive. Do not rewrite its scores.

Add an explicit optional decoder mode that clips a context token at the source
boundary only when every character outside that boundary is whitespace. Keep
ordinary prefix and suffix text ineligible. Validate original offsets before
clipping and preserve the exact source substring and character positions. The
default remains the original strict rule so historical decisions reproduce.

Re-decode all 291 stored logit pairs under this one change. Preserve the model,
questions, candidate order, tokenization, logits, null margin, maximum span length,
literal-content guard and labels. No model calls, retrieval calls, threshold
tuning or additional data are permitted. Reconstruct decisions independently
and report both versions, including regressions. This is a post-hoc development
correction prompted by the original errors, not a fresh model evaluation.

Freeze this protocol and corrected decoder before calculating the new scores.
Retain 153 query observations from 51 unique questions, distinguish repetitions,
and keep failed runtime attempts separate from valid abstentions. Preserve the
complete qualified fact as context; a selected literal span remains a model
judgment, not proof of temporal applicability or conflict resolution.
