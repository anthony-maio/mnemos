# Local support input boundaries: admission controls passed

The tokenizer-only suite completed all eleven controls: seven admitted and
four unavailable, matching the declared policy. Exact 8,192-token complete
native framing and exact 64,000-byte payloads were retained. Inputs beyond
those limits were rejected whole. No model forward, training or hosted call
ran. This is an inactive candidate check, not production behavior, semantic
acceptance, latency qualification or a pilot result.

| Boundary | Admitted control | Rejected control |
| --- | --- | --- |
| Candidate inventory | Zero, one, four and five facts | Six facts |
| Owned source inventory | Eight sources per fact | Nine sources on one fact |
| Complete native frame | 8,192 tokens, including boundaries and decision markers | 8,193 tokens |
| Payload | 64,000 bytes, with all owned excerpts | 64,001 bytes |

The full-token and byte controls each retain five facts and eight sources per
fact. The byte-bound frame uses 5,029 tokens. Stress filler is deliberate and
does not make these independent semantic examples. The eleven controls reuse
six unique synthetic fixture facts; they are outside training and reserved
semantic partitions. Existing 64-case, 160-target preparation remains unchanged.

There are ten answerable control requests and 38 supported target opportunities.
All four unavailable controls are answerable, accounting for 17 opportunities.
Those requests and targets remain in coverage and recall denominators. Missing
or duplicate observations fail accounting. Precision and semantic recall remain
null because the suite emits no classifier decisions; unavailability is not
credited as semantic abstention.

The new admission wrapper reuses the historical strict serializer and complete
native frame, then enforces the explicit eight-source maximum. It does not
truncate, drop sources or return a partial cohort. Historical source and timing
records are unchanged. The wrapper is not wired into live retrieval, and the
previous computation pass does not performance-qualify every combination within
these limits or this added admission work.

Before execution, review found two harness gaps: eager package imports would
load unrelated dependencies, and an interrupted run could lose completed
observations. A standalone sibling loader now avoids application imports.
Atomic observation files and failed terminal receipts retain completed frames,
the active control, unattempted IDs and expected denominators. Regression checks
cover a site-disabled entry point, tokenizer failure partway through the suite
and failure during terminal validation. All 35 focused checks pass.

The actual run used committed source
`9130a0ff04bf9910b21b9fabeda5ff4b4d804422`, the unchanged 25-package model
environment and the existing frozen local tokenizer. All seven bound source
files, tokenizer assets, dependency version inventory and prepared cases matched
after execution. The process exited zero in 7.21 seconds. That elapsed setup and
tokenization time is not an inference or recall-latency measurement.

The offline audit reconstructed all seven full frame fingerprints, complete
payloads, owned fixture support and accounting totals, and rechecked both frozen
semantic data receipts. It did not rerun tokenization. The rejected 8,193-token
count is supported by the committed preparation assertion and frozen execution
evidence. Independent terminal review verified all 38 owned-support labels,
seven frames, five artifact hashes, seven source bindings and five tokenizer
file hashes. It independently verified the 64,001-byte overflow, but did not
retokenize the rejected token case. A separate review ledger binds labels to
the original request hashes and target IDs; raw cases remain unchanged.

Private evidence is in `.venv/local-support-v2-boundary-controls-v1`, with full
raw cases, frames, observations, freeze, terminal result and offline audit.
The private archive has 21 members and 55,802 bytes; every included file verifies
on readback. SHA-256 is
`6462a834cb1cc75d5d437a7955b31364acd6c2fc95234b3bd200abaf828846ca`.
Tokenizer assets and dependency bytes remain external. The initial archive
lookup used Windows separators; normalizing the stored member names verified
the existing archive without rewriting it.
Tokenizer binding is
`f3548af10ef3685970b6690ceb1e47509cb8caf650084fdcc19b2b5974895c83`.
This closes the mechanical boundary check only. Sparse reserved semantic
coverage, supported evidence counterfactuals, a reviewed numerical recipe and
the bounded backward probe remain before learning. Original quality, complete
trained timing, capture, fresh acceptance, native/pilot and delivery gates remain.
