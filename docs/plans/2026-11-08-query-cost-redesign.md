# Query-time cost redesign constraints

The closed v2 run and its output-cost audit rule out assuming that source caching plus an identifier fix is enough. The observed query generation alone takes 4.59-4.84 seconds. No new semantic run is justified solely by replacing ambiguous indices with long explicit references.

A prospective design must address two separate problems:

1. Source-side correctness: retain prerequisites in structured conditions, distinguish support for an attributed report from proof of the reported event, and prevent fabricated source locations. Reduce repeated quotation generation without relaxing exact-source validation. Any asynchronous annotation path must preserve unavailable/backlog status and original cohort visibility.
2. Query-side cost: avoid making the model regenerate source text, long digests and redundant metadata. Reduce prompt duplication and reply length while retaining all eligible candidates, unresolved/conflicting evidence and explicit temporal intent. Measure tokenizer counts and actual complete retrieval cost, including misses and required annotation work.

The explicit full claim references remain useful internally. A future wire format can map short request-local labels to those immutable references before inference. The owner must bind the complete snapshot, source inventories, policy and runtime to the transport request and revalidate before and after execution. Short labels alone cannot authenticate a response or authorize a fact. Duplicate, missing, foreign and stale references must still fail closed. Moving a digest outside generated text must preserve trusted request association and the audit trail; merely deleting validation is not a cost fix.

Do not collapse uncertainty into negative support to reduce output length. Do not omit competing candidates or source qualifications to fit a smaller prompt. Do not reinterpret source-state answers as current validity. Any compact categorical encoding needs an explicit unavailable state and exact whole-cohort coverage, with every decoded outcome traceable to the original fact and own evidence.

Prior single-token and discriminative experiments already failed semantic quality. Their results remain relevant: shorter output is not sufficient, and a new wire representation must not be presented as a semantic improvement without new evidence. Source-side correctness and query interpretation need separate grading before an integrated fresh acceptance run.

No replacement model, prompt, schema, threshold or runtime has been selected here. This document records the measured constraint and requirements for the next implementation; it does not waive any original goal requirement or reopen a closed screen.

## Compact wire boundary

`compact_query_wire.py` implements a preparation/decoding boundary without a provider call or final support decision. Preparation requires the exact current snapshot and complete source inventories. It retains all original facts, own-source excerpts and claim annotations in model-visible input, replacing identity metadata with explicit short labels such as `c0`. Full references, scope, source/query identities and the complete snapshot remain in an immutable owner-side request record. Its digest binds the input and alias map; a fresh request ID distinguishes even otherwise identical invocations.

The model response uses `t` for temporal intent (`S` source-state, `D` date, `U` unresolved), `d` for an ISO date or null, and `c` for the complete alias-to-code mapping. Each two-letter code retains relation match (`M`, `X`, `U`) and condition applicability (`A`, `X`, `U`). No model-generated digest or fact/source identifier is required. For example, `{"t":"S","d":null,"c":{"c0":"MA","c1":"UU"}}` represents one matching/applicable claim and one unresolved claim; it is not itself a selected answer.

The trusted transport must attach request ID and binding digest from its actual invocation, never copy them from model text or infer them from short aliases. The decoder verifies those fields, the unchanged owner record, complete unique alias coverage and live snapshot fences. It rejects duplicate JSON keys rather than allowing the parser to overwrite an earlier judgment. Decoding produces full referenced claims and temporal intent; a caller must still apply source support, temporal validity, conflict and unavailable rules. Request hashes are integrity bindings, not cryptographic authentication of an untrusted transport.

The compact boundary, explicit-reference and temporal suites passed 46 tests in 2.08 seconds, retained at `.venv/compact-wire-v1.xml`. Tests cover all nine relation/condition states, dated intent, foreign/missing aliases, duplicates, cross-request replies, altered request records, incomplete transport completion, source revocation and attempted competitor omission. These are model-free checks. No tokenizer, semantic accuracy, complete latency or production claim is established; no existing frozen implementation was modified.

## Transport and decision integration

`compact_query.py` now supplies a strict schema for the exact request-local labels, the complete model-facing prompt, a trusted-provider adapter and the final decision step. The adapter attaches request identity from its own invocation after receiving confirmed complete text; no generated identity is accepted. The caller still owns runtime verification, tokenizer checks and durable request/reply logging. The asynchronous entry point retains the 15-second cooperative limit, no retry and live fences before and after transport. Cancellation propagates and ordinary failures return unavailable.

The evaluator consumes all decoded judgments and preserves the v2 source-support, temporal, conflict and unavailable rules. Selected facts retain their per-claim temporal basis. No negative or missing judgment becomes successful abstention merely because its encoding is shorter.

The compact adapter/evaluator and related boundary suites passed 133 tests in 2.74 seconds, retained at `.venv/compact-query-v1.xml`. Exhaustive parity checks cover all 81 relation/condition combinations for two claims against the frozen v2 evaluator, comparing both decisions and selected support bases. Separate cases cover conflicting values, unsupported and unresolved whole facts, dated requests against unanchored assertions, local transport identity, incomplete replies, source revocation and cancellation. This is deterministic decision parity, not model semantic parity.

Next measure the actual full prompt and representative response token counts before choosing a new execution policy. A runtime wrapper must journal the full owner request record and actual provider wire reply, not just the compact model text. Keep source annotation quality work separate. No compact semantic run, full latency claim, new default or production integration is established.
