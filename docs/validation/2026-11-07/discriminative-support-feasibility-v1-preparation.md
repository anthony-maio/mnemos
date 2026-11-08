# Discriminative support screen preparation

The [prospective protocol](../../plans/2026-11-07-discriminative-support-feasibility-v1.md)
was committed at `29e34cfd` before input preparation. Preparation succeeded on
its first attempt. It produced 68 known-question records containing 123 support
pairs and 118 directed contradiction pairs, with at most nine pairs per question.
Expected IDs are retained separately from classifier inputs.

The exact pinned revision's card, configuration and tokenizer metadata were
downloaded and hashed. Configuration confirms standard
`DebertaV2ForSequenceClassification`, no custom `auto_map`, and the label mapping
0=entailment, 1=neutral, 2=contradiction. The pinned card declares Apache-2.0.
These checks establish input compatibility for the proposed screen, not model
quality or inference performance.

The private preparation manifest is
`.venv/discriminative-support-feasibility-v1/preparation.json`. It records source
URLs, metadata hashes, original observation hash and separate input/grading
hashes. The prepared inputs hash is
`2cedf7a2bee4b6aa8bf5a5be6ec2e3ec9c75b8a4951e6da5fdcdc22488788655`.

No weights were downloaded and no inference ran. Token limits, isolated runtime,
decision-logic validation, artifact freeze and measured execution remain pending.
There is no passing support, latency, fresh acceptance or pilot result here.
Product source, live configuration, databases and earlier scores remain unchanged.
