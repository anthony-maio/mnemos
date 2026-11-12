# Direct claim readout in the full context

The [completed local training attempt](../validation/2026-11-12/local-support-learning-v2.md)
is a closed semantic failure. Its initial and final models selected the last
candidate in every tuning/development case. Do not rerun that configuration,
change its threshold, alter its data or transfer its computational admission
to another representation.

Trace the failure before any new experiment. The original frame stores all
candidate text and owned evidence, then appends indexed decision markers. The
shared scalar classifier reads the final newline in each marker. All 354
corpus readouts use token 187, but their contextual hidden states need not be
identical. The original checkpoint's configured NLI path uses mean pooling,
then its trained dense/norm head and three-way classifier. The completed
attempt intentionally discarded those four head tensors and trained a fresh
linear scalar head on the marker states. Correct gradients and index binding
do not prove that this new task learned support. None of these observations
establishes the internal cause of the positional pattern.

## Selected mechanical change

Bind each readout directly to the complete candidate claim, while retaining
the entire query, date, cohort, source IDs and owned excerpts in one encoder
input. Record claim and owned-source character ranges as canonical JSON is
written. Repeated claims, quotes and source text cannot redirect those ranges;
substring search is excluded. Preserve the exact canonical payload bytes from
the previous format. Remove its appended decision markers and introduce the
explicit claim-span format preamble.

Map each complete claim range through the pinned tokenizer's actual offsets.
Ranges are half-open character positions in preamble plus serialized JSON,
excluding string quotes; JSON escapes remain serialized. Include every token
overlapping a value, including a token crossing its quote boundary. Native
CLS/SEP add one token-index offset and have no character offsets. Reject
invalid, empty, incomplete or ambiguous mappings. Every non-whitespace
character of the serialized claim, query, reference date and owned source
must be covered. A token overlapping two semantic fields is ambiguous.

The first tokenizer audit exposed NFC composition: offsets on original text
can omit a combining accent even though the composed character is encoded.
Preserve the raw payload and its raw character ranges. Also record explicitly
NFC-normalized model text and model character ranges, mapping token offsets
in that normalized space. Verify normalization cannot cross a semantic field
boundary and require complete non-whitespace coverage there. The pinned
tokenizer already applies NFC; an actual-tokenizer audit must prove this
explicit normalization preserves its native token IDs. Do not skip arbitrary
missing characters or normalize the stored source payload. Preserve the first
failed tokenizer audit separately from subsequent corrected evidence.

Average all mapped claim hidden states in float32, preserving ID/version and
candidate order, then apply one shared scalar head. This contract exposes
the claim's own contextual states instead of requiring an arbitrary tail
index to dereference it. It does not select a new initialization, training
recipe, threshold or model execution. Empty cohorts preserve the request and
bypass head computation. The pooling helper accepts only trusted frames;
external reconstructed frames require binding verification before use.

Keep the whole request within 8,192 native tokens, 64,000 payload bytes,
zero through five facts and at most eight complete owned sources per fact.
Reject whole inputs beyond these limits. The new preamble and removal of
markers change token counts, so old token-overflow control admissions cannot
be transferred. Audit their new counts separately and derive exact new 8,192
and 8,193-token controls without modifying historical inputs or labels.

## Alternative and limits

Retaining the pretrained NLI classifier with one complete-context pair per
target is a different substantive option. It preserves trained entailment
weights, but its entailment objective still differs from query relevance and
fully owned support. It also introduces repeated complete-context work and
an unmeasured batching/memory/latency burden. Batching does not imply exactly
fivefold latency, and the previous single-pass result cannot qualify it.
The direct-span contract is the smaller change to verify first.

Claim pooling still permits the contextual encoder to attend to other facts'
sources. It cannot enforce evidence ownership by itself. Mean pooling can
dilute a small decisive clause or negation even when every token is included.
Permutation changes can still affect contextual states. These remain semantic
risks to measure; structural correctness is not their resolution.

## Verification and next boundary

Verify duplicate claims, repeated quoted source text, escapes, Unicode,
candidate reordering, long final conditions, native boundary tokens, complete
source inventories, all byte/token edges and empty cohorts. Deterministic CPU
hidden states must prove complete FP32 pooling, shared-head ordering, gradient
flow through every mapped claim token and nonfinite rejection. A tokenizer-only
audit must check all 116 known corpus requests against the old payloads and
source ownership, plus historical and new boundary controls. It must preserve
all old data, labels, scores, source and artifact receipts.

This increment stops at the inactive serialization/readout contract. No
pretrained encoder inference, optimizer update, hosted call, live install or
semantic/timing claim is included. A subsequent learning decision needs its
own reviewed numerical and execution contract. It must defeat the observed
position-only behavior and meet every original quality gate; fresh acceptance,
trained complete timing, capture, native, the actual two-week pilot and delivery
remain separate requirements. Local operation stays required and Jev stays
explicitly optional, with no automatic fallback.
