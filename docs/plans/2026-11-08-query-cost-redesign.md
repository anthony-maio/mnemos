# Query-time cost redesign constraints

The closed v2 run and its output-cost audit rule out assuming that source caching plus an identifier fix is enough. The observed query generation alone takes 4.59-4.84 seconds. No new semantic run is justified solely by replacing ambiguous indices with long explicit references.

A prospective design must address two separate problems:

1. Source-side correctness: retain prerequisites in structured conditions, distinguish support for an attributed report from proof of the reported event, and prevent fabricated source locations. Reduce repeated quotation generation without relaxing exact-source validation. Any asynchronous annotation path must preserve unavailable/backlog status and original cohort visibility.
2. Query-side cost: avoid making the model regenerate source text, long digests and redundant metadata. Reduce prompt duplication and reply length while retaining all eligible candidates, unresolved/conflicting evidence and explicit temporal intent. Measure tokenizer counts and actual complete retrieval cost, including misses and required annotation work.

The explicit full claim references remain useful internally. A future wire format can map short request-local labels to those immutable references before inference. The owner must bind the complete snapshot, source inventories, policy and runtime to the transport request and revalidate before and after execution. Short labels alone cannot authenticate a response or authorize a fact. Duplicate, missing, foreign and stale references must still fail closed. Moving a digest outside generated text must preserve trusted request association and the audit trail; merely deleting validation is not a cost fix.

Do not collapse uncertainty into negative support to reduce output length. Do not omit competing candidates or source qualifications to fit a smaller prompt. Do not reinterpret source-state answers as current validity. Any compact categorical encoding needs an explicit unavailable state and exact whole-cohort coverage, with every decoded outcome traceable to the original fact and own evidence.

Prior single-token and discriminative experiments already failed semantic quality. Their results remain relevant: shorter output is not sufficient, and a new wire representation must not be presented as a semantic improvement without new evidence. Source-side correctness and query interpretation need separate grading before an integrated fresh acceptance run.

No replacement model, prompt, schema, threshold or runtime has been selected here. This document records the measured constraint and requirements for the next implementation; it does not waive any original goal requirement or reopen a closed screen.
