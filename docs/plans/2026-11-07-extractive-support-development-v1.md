# Extractive answer-support development screen

Test whether a dedicated question-answering model can filter related facts that
do not answer a query. This is a development screen over existing frozen hybrid
retrieval results. It does not rerun retrieval, evaluate extraction, open the
unused fresh corpus, or qualify the daily-use pilot.

Use `deepset/deberta-v3-base-squad2` at revision
`eea39c60cc305c2e4a9504f5ff117294bebb42db`. The publisher describes training on
SQuAD2, including unanswerable questions, and licenses the model under CC BY 4.0.
Keep the model card and verified asset hashes with the private evidence.
[Model card](https://huggingface.co/deepset/deberta-v3-base-squad2)

Run the safetensors weights locally with PyTorch 2.10.0+cpu, Transformers 4.57.6,
safetensors 0.8.0, SentencePiece 0.2.1, four CPU threads and one interop thread.
Use float32, evaluation mode, inference mode and deterministic algorithms. Pin
seed 2026110702. Disable network model loading and remote code. These are private
evaluation dependencies; do not add them to the package or pilot installation.

Reuse the first hybrid repetition of the accepted-state seven-query development
screen and all 44 hybrid inputs from contrast-support v6. Verify their exact
candidate contents, assertion status and retained excerpts against the original
declared fixtures. Preserve candidate order and original snapshot hashes. These
are previously measured inputs, not new retrieval runs or 51 independent projects.
Keep expected IDs in a separate file that the inference loop does not read.

Evaluate three shuffled repetitions of these 51 queries: 153 query observations,
291 query/excerpt pairs, and 48 skipped candidate occurrences marked conflict or
unknown. Each candidate in these fixtures has one retained excerpt. Unknown and
stored-conflict candidates are never selected. Scope eligibility comes from the
frozen, verified retrieval inputs; this screen is not a new live scope check.

For each eligible candidate, pair the literal question with this context:

```
Reference date: {original reference date}
Retained source excerpt:
{exact retained excerpt}
```

The claimed fact is withheld from the model to avoid priming an unsupported
claim. Only tokens wholly inside the retained excerpt can supply an answer.
Question, date prefix, special and padding tokens are ineligible. Use at most
64 question tokens, 512 total input tokens and 30 answer tokens. Never truncate;
oversized or malformed input is an operational failure. Keep complete input IDs,
character offsets, masks, sequence IDs and raw start/end logits for audit.

Choose the contiguous eligible span with the largest start-plus-end logit sum.
Break span ties by earliest start and then earliest end. Select it only when its
score strictly exceeds the CLS start-plus-end score, with fixed null margin zero.
A CLS tie abstains. This follows the span/null comparison used in the official
Transformers QA implementation, with explicit source-region restrictions.
[Transformers QA source](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/pipelines/question_answering.py)

Select a candidate fact only if the chosen source span also occurs literally in
its full content. Retain the complete qualified fact as selected context; the
span is evidence for the decision, not a replacement memory. Literal overlap
and model scores do not establish entailment, resolve unmarked conflicts, or
prove temporal applicability. This independent-excerpt method does not compare
candidate claims. These limitations must remain visible in the results.

Count expected-ID hits, clean hits without extra IDs, unsupported selections,
correct empty-result abstentions and operational failures separately. Failure
cannot earn a hit or abstention. Report each suite separately and distinguish
unique questions from repetitions. Compare the 44-question suite descriptively
with its existing joint and contrast judgments using the same frozen hybrid
candidates. Do not infer native-host latency from local CPU measurements.

The total run deadline is 600 seconds, checked before each pair. Preserve any
errors and all attempted inputs; do not silently retry. No threshold tuning,
label changes or fresh-corpus runs follow from a negative screen. Save the clean
source commit, runner hash, dataset hashes, dependency versions, model hashes,
input hashes and outputs. Record model load time separately from pair and query
execution. Audit results from raw logits without another model call.

Frozen unlabelled input SHA-256:
`87af4a70b776e56eedcb8ae9a5cb56244ddcddae2da3403e98fb2db4cce9b195`.
Frozen labels SHA-256:
`07efc92860812ac9848dbcb91c1b24f1d53af50d08431b92500cab510a8f89b1`.
The private runner hash is recorded in the run control before inference. No live
configuration, memory database, package dependency or production default changes.
