# Qwen3.5 qualified inventory v4 semantic review

All 46 original development cases have descriptive review entries. Runtime
outcomes: {'runtime_valid': 40, 'error': 6}. This candidate is not adopted. Provider failures and
validation rejections remain separate from successful abstention; inspect
runtime-errors.json for the final-response counters, without treating a rejected
inventory as a retained memory. Original rubric ambiguities remain attached.

The current output preserves the mixed marker separately from temporary tool
restrictions. Some cases now return valid empty retained lists, but semantic
failures remain: N02 treats thinking out loud as an explicit retention request;
P01 stores classification commentary as preference content; P06 upgrades possible
future telemetry acceptability into explicit permission. Source-exact quotation
and schema validity do not establish meaning or durable retention intent.

P04 preserves the historical-report warning in a separate companion assertion.
Several policy cases omit qualifications or split them across assertions, and
no downstream co-retrieval or target reconciliation was tested. Original rubric
ambiguities and all rejected outputs remain preserved, not relabeled as passes.

This is one reused development repetition with thinking disabled and a different
provider route from v3. It does not isolate the effect of thinking, establish
fresh held-out accuracy, or justify a production configuration change.
