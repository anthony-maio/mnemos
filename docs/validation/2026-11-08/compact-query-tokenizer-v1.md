# Compact query tokenizer comparison

The retained Qwen3:14b tokenizer processed 255 items without semantic generation: three original query prompts, three compact prompts using the same actual source inventories, three recorded reply bodies, three minified versions of those bodies, and 243 synthetic compact replies covering all two-claim relation/condition combinations across the three temporal intents.

| Representation | Items | Token range | Median |
| --- | ---: | ---: | ---: |
| Original rendered prompts | 3 | 2,312-2,322 | 2,316 |
| Compact rendered prompts | 3 | 694-697 | 696 |
| Recorded original reply bodies | 3 | 188-199 | 197 |
| Minified original reply bodies | 3 | 134-145 | 143 |
| Synthetic compact reply bodies | 243 | 21-32 | 21 |

Paired prompt reductions were 69.95%-69.98%. The compact inputs retained the original candidate facts, own excerpts and source annotations; identity metadata stayed in owner-side records. The compact prompt also changes instruction wording and schema, so token reduction does not establish semantic parity. The three source snapshots come from the completed prefix of a failed development run, not the full cohort or fresh acceptance.

Reply bodies were tokenized without special tokens or prompt context. Their counts can differ from actual generation: the original provider reported 189-200 generated tokens for these replies, compared with 188-199 standalone body tokens. Synthetic compact replies are wire-state enumerations, not model answers. They were not used to repair, replace or regrade the original responses.

At the earlier observed query generation rate of about 41 tokens/second, even 21 tokens corresponds to roughly 510 ms before prompt processing, retrieval or transport. That is an illustrative extrapolation, not a measurement of the compact path. It gives no basis to claim the full 500 ms target passes. Supported facts per token and semantic recall remain unmeasured because no compact semantic responses were generated.

Wrapper session 23075 ended with exit code zero. The model was unloaded, parent/native processes were absent, both ports were closed and runtime/input bindings were unchanged. The independent audit verified all 255 ordered journal records, exact content hashes, tokenizer options, counts and paired comparisons. Detailed counts and evidence hashes are in `compact-query-tokenizer-v1.json`; raw inputs and owner records remain in `.venv/compact-query-token-inputs-v1`, with runtime evidence in `.venv/compact-query-tokenizer-v1`.

The next design decision must account for both the remaining output budget and unresolved source quality. This result supports a smaller representation only; it does not justify promotion, a deadline change or reopening the failed v2 screen.
