# Source-only assertion inventory preparation

This freezes 46 reused development cases before generation with Qwen2.5:14b,
think omitted, temperature 1.0, 8192 output tokens and 180-second timeout on
local Ollama. Exact model digest and source hashes are in preparation.json.
The candidate module is bound to 60f79ca; provider/runtime code uses the same
cc940ac snapshot as the preceding comparison. The candidate never sees the
existing unrelated control fact and cannot propose support or correction.

Sixteen deterministic candidate tests passed, Black passed, and Python 3.10
mypy passed. Offline preflight checks all 46 source schemas, prompt isolation
and source-ID binding with zero generation calls. Empty stub inventories are
wiring checks, not successful extraction or evidence of abstention quality.

All original semantic rubrics and ambiguity reviews remain unchanged. Review
independent fact retention, unwanted retention, attribution and unsupported
interpretation separately. Runtime quote/ID validation is not semantic proof.
The source-only diagnostic does not validate target reconciliation, support,
correction or durable service writes. Neither the production prompt nor the
service contract is changed. No fresh evaluation corpus is exposed.

At this preparation checkpoint the model run is NOT STARTED. Consult a later
terminal archive for outcomes; never treat this preparation as a result.
