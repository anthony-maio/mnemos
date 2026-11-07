# Compact retrieval response measurement v1

Status: prospective. Freeze source and this protocol before measurement.

The full MCP recall response repeats literal fact content in its fact records
and context string, includes source excerpts and retrieval scores, and uses
indented JSON. The opt-in compact format retains the exact rendered context and
source references, deferring excerpts and scores to inspection. It does not
change retrieval, support policy, fact content or the default response.

Use an isolated SQLite service and the actual stdio MCP transport, with no
model inference or live profiles. Seed explicit synthetic user facts with
source evidence. Include production/staging qualifications, negation, quoted
Unicode text, two conflicting facts, a separate captured-conversation record,
and a forbidden fact in another project. Run six cases: one result, up to five
results, zero requested results, captured evidence, an injected embedding
outage and conflict recall. Keep the corpus and allowed scopes fixed.

Run full and compact for each case, three repetitions, alternating their order
by repetition. This gives 18 pairs and 36 retrieval calls. Both formats must
return the same literal context, record IDs, versions, scopes, assertion status,
source references and source-availability status. Keep degraded diagnostics and
captured-evidence warnings. Reject any foreign-project content. Verify stored
fact content and evidence remain unchanged, apart from retrieval access metadata.

Record actual text replies and complete MCP result objects, UTF-8 bytes and
roundtrip latency. Also serialize full data without indentation and compact data
with indentation to distinguish projection savings from JSON whitespace savings.
These two derived representations are not extra transport calls. Use byte
measurements honestly; they are not exact Codex or Claude token counts.

Inspect each returned reference once per pair. Verify its current content and
complete evidence match the full response. Report the additional reply bytes
and latency separately, including compact recall plus inspection. Do not claim
that every workflow becomes smaller or faster. Timing is a local transport
measurement with three repetitions, not a native-host model benchmark.

Keep context-budget metadata unchanged because it describes rendered context,
not the whole transport envelope. Count returned facts as preservation of the
full response's selection, not semantic answer-support precision or fresh recall.
No source-backed semantic compression or new tokenizer compatibility is implied.

Bind all source hashes, fixture values, protocol and original responses.
Independently reconstruct context equality, reference preservation, byte totals
and latency aggregates. Stop the owned service and child MCP process, verify the
endpoint is closed and preserve the private evidence archive. Record the installed
Codex/Claude capability inspection separately. No activation, publication or
pilot day occurs.
