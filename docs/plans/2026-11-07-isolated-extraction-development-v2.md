# Isolated extraction development v2

Status: prospective regression check after the v1 decoder-schema failure.
Commit this protocol and the repair before inference. Preserve v1 unchanged.

Repeat the [v1 protocol](2026-11-07-isolated-extraction-development-v1.md) on the
exact same 16-case corpus, SHA-256
`9a714ad692fe0e8a4e8e04b02386c8eb194fba59cff0d501f6d217c227563bfe`.
Keep its labels, scorer, model digest, native tokenizer, options, shuffle seed,
one repetition, deadlines, input/output limits and separate guard denominator.
This is development regression evidence, not fresh evaluation.

The only production change splits each user source's decoder alternatives by
kind. A continuing requirement permits either revision decision. Every other
kind requires `revises_existing_requirement: false`. Non-user source kinds
remain restricted and cannot gain revision authority. Each alternative carries
its complete object constraints so a decoder need not enforce sibling schema
constraints. Runtime validation remains unchanged, including for providers
that ignore structured output.

The repair changes the supplied schema, not classification instructions,
reconciliation, facts, targets or thresholds. Compare the actual original input
hashes and classification prompts across runs. Schema size may increase; count
the exact new prompt-plus-schema input before each inference. Retain every
case, including operational failures or new regressions. Passing 16 cases would
still leave sample size, fresh evaluation and broader semantic quality open.

Independently reconstruct the new results and compare the original expected
outcomes. Specifically verify the correction's source kind, false revision
flag, target identity and current version. Report continuing-requirement and
non-user outcomes too, without treating contract tests as model quality.
Preserve original latency and token accounting for both runs. Unload the model
and stop only the owned runtime when finished. No live pilot or release action
is included.
