# Direct claim readout contract

The inactive claim-span serializer and pooling contract pass their model-free
checks on source `588548996637d3efbbfbf47faba9fe9f6eacdfc9`. Ten focused
contracts and a pinned-tokenizer audit verify all 116 known corpus cases and
354 claim mappings. Original canonical payloads are byte-identical: complete
queries, dates, candidates, IDs, revisions and owned sources remain present.
This is a mechanical representation change, not a semantic result.

## What the failure investigation established

The prior [failed learning result](local-support-learning-v2.md) stays closed.
Independent decoding found no candidate-order or tail-marker binding mismatch
in its 116 saved corpus frames. Its 354 classifier positions all point to
newline token 187. The initial model selected only the last candidate in all
116 cases; the trained model did so in 39/56 training cases and all 31 tuning
and 29 development cases. Its other 17 training cases added selections while
retaining the last candidate.

A repeated token can have different contextual hidden states. These findings
do not prove an internal cause. They do establish that the frozen training run
failed its intended semantic task. The original checkpoint's mean-pooled NLI
head was intentionally discarded; the new scalar head had to learn support
from appended index-marker states. The new contract instead binds each readout
directly to every token in its complete claim, in one full-context input.

## Verified contract

Character spans are recorded while canonical JSON is written, rather than
located by substring search. Each claim keeps its ID/version and separate
owned-source ID/event/span records. Token mappings include every overlapping
token, retain native CLS/SEP offsets and reject ambiguous mappings across
semantic fields. The synthetic tensor checks verify FP32 averaging, shared
scalar-head order and finite outputs, with gradients through every mapped
claim token and no pooling gradient through unrelated tokens. Empty cohorts
preserve their request and bypass head computation.

The first native-tokenizer audit failed on a combining Unicode accent. The
tokenizer's NFC normalizer composes alpha plus U+0301, while its offsets on
original text omit that combining code point. That first failure and its
source remain in a verified seven-member private archive, SHA-256
`ebf6485c42f58d6bfd791fd954ebb5ede18a5a5ff5ddc2306c76214966f990b3`.

The correction preserves raw payload/text/spans and records explicit NFC model
text and model spans. Normalized field boundaries must match each separately
normalized raw field. Missing non-whitespace coverage still fails. Actual
token IDs from explicit NFC equal the pinned tokenizer's IDs on raw text for
all 116 corpus cases and the Unicode/escape/repeated-text fixture. This is a
verified property of those inspected inputs with this pinned tokenizer, not
a guarantee for arbitrary tokenizers or unseen Unicode.

The known corpus uses 264-979 tokens per new complete frame and 6-33 tokens
per claim readout. Eleven historical controls were reframed separately:
eight admit and three reject whole. The previous token-overflow request now
uses 8,155 tokens because the format removes decision markers; its historical
rejection remains unchanged. The former exact-token request uses 8,154 tokens.
New derivative controls establish exactly 8,192 admitted versus 8,193 rejected
tokens without deleting original source text. The exact 64,000-byte request
admits at 4,991 tokens; byte overflow, nine sources and six facts reject whole.
These repeated controls provide no new independent semantic examples.

There are 126 saved frames: 116 corpus, eight admitted historical controls,
one new exact-token frame and one Unicode fixture. All 115 frozen files and
25 runtime versions remain unchanged. Frame hashes were independently
recomputed, and the archive was read back byte-for-byte. It contains 64 members
and 2,650,561 bytes, SHA-256
`b7e18cf3affee2c53905eea0a648b08d8592b71825c03de3ee8a849cad577c31`.
The [receipt](claim-span-readout-v1-receipt.json) includes individual frame
hashes and changed historical admissions. Large original model/runtime files
are hash-bound; no new encoder or trained artifact was constructed.

## Limits and next work

The pooling helper requires a trusted serializer frame. It does not authenticate
arbitrary JSON-reconstructed frames. Contextual attention can still mix other
facts' evidence, mean pooling can dilute decisive qualifiers, and candidate
reordering can still affect contextual states. Structural ownership records
do not prove semantic source noninterference or full-clause support.

The [design decision](../../plans/2026-11-12-claim-span-readout-design.md)
permits this inactive contract only. A learning attempt needs a separately
reviewed numerical and execution design. No pretrained encoder forward,
optimizer update, hosted call, new quality grade or latency run occurred here.
The previous capacity pass cannot qualify this representation. Product
defaults, local host configuration, live databases and both project pilot
files remain unchanged. Local operation is required and Jev remains explicit.
All original quality, fresh acceptance, complete timing, capture, native,
two-week pilot and delivery obligations remain open.
