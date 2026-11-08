# Recorded generation cost in the closed v2 screen

An offline audit of all 14 completed provider replies separates generation from prompt processing. It reads the original journal at SHA-256 `450b414838fa1dc1c0c58b6f51729fc10fbbefa025378381a008b8009869000e`; no model or tokenizer calls were made and no result was regraded. The timed-out request has no complete provider timing and is excluded from these component summaries.

| Completed replies | Generated tokens | Generation time | Median provider total | Median prompt processing |
| --- | --- | --- | --- | --- |
| 11 source replies | 243-589 | 5,806.77-14,191.11 ms | 6,565.67 ms | 185.62 ms |
| 3 query replies | 189-200 | 4,593.78-4,837.16 ms | 5,709.22 ms | 606.78 ms |

Median generation throughput was 41.50 tokens/second for source replies and 41.14 for query replies. At that observed rate, 500 ms would accommodate about 21 generated tokens only if every other cost were zero. That arithmetic is a diagnostic scale comparison, not a forecast for a changed schema, model, hardware setup or warmed request. Median component times cannot be added to manufacture a total percentile.

The query replies already exceeded the full retrieval target by more than nine times in generation alone. Median prompt processing also exceeded 500 ms in this completed prefix. The reuse pass never ran, so there is no measured warmed-query improvement. Source caching does not remove query generation, and the explicit-reference repair does not address this cost by itself.

Full hexadecimal request and claim identifiers, repeated field names, source quotations and normalized text all require generated output. The audit records actual reply bytes and hypothetical minified JSON bytes, but byte reduction is not a token or latency measurement. Do not claim that minification or shorter references will meet the target without measuring the complete path.

The next candidate needs a different query-time representation or execution strategy, while preserving complete candidate visibility, evidence binding, condition/time judgments, conflict handling and unavailable outcomes. Source annotation quality remains independently unresolved. A shorter query wire format cannot repair omitted conditions, unsupported report assessments or invented quotations.

Per-reply provider counts, event digests and summaries are retained in `relation-inventory-output-cost-v2.json`. Original full quality, 500 ms retrieval, 100 ms capture and actual pilot requirements remain unchanged.
