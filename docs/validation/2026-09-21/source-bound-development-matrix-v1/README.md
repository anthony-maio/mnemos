# Seven-profile development matrix

All seven profiles completed one automatic-capture development session without
runtime errors on frozen source a6b0bf3. Local Qwen3.5 9B and Nomic embeddings
were used with source_bound extraction, temperature 0, think=False, 8192 output
tokens and a 180-second provider timeout. Exact digests and code/input hashes
are recorded. No held-out scenario was executed in this run.

All seven original exact-text answerable-hit scores are false. The full profile
nevertheless retrieved the requested rule with its original introductory phrase:
"For this diagnostic and release-planning work:". The gold text omits that
phrase. This is a wording mismatch under the frozen exact-text criterion; the
raw score is retained unchanged. The full profile also retained an additional
system-context memory, which is not a requested gold fact. Both observations
must remain visible rather than reporting runtime completion as quality success.
All seven foreign-profile no-answer checks passed. No aggregate quality claim
is supported by one session and one repetition. No independent semantic score
or confidence interval is claimed.

The run does not show useful execution or superiority of every module. Baseline
has real embeddings but no observed generation. Original provider responses,
operations, provenance and exact-text judgments remain in each profile artifact.
