# Two-stage qualification selector

Implemented an experimental selector using query-independent source qualification followed by fresh query-time answer judgments. Selected witnesses must pass both stages. Every query-time request retains the complete original comparison cohort and its source annotations; rejected and unresolved sources are still visible.

The source batch is committed only after all missing qualifications succeed. Provider errors, malformed responses, revision changes, and cancellation propagate; they do not become successful abstention. A later answer failure prevents an earlier positive witness from being returned. Role delimiters and the complete request/byte budget are checked before provider work. Exact model token limits and the shared deadline remain provider/caller responsibilities.

Validation: 40 tests passed across the new selector and qualification records, including 12 new selector cases. The checks cover reuse across different queries without answer reuse, source rejection and uncertainty, source support with a negative answer, batch failure without partial caching, later answer failure, preflight rejection, and policy mismatch. Both changed Python files were formatted with Black. These are model-free unit checks; no semantic or latency benefit is established.

No model calls, production defaults, databases, or host configuration changed. Next: author fresh development inputs with separate source and answer labels, prepare exact cold/warm requests, and freeze the actual runtime before inference. Earlier failed cohorts remain closed.
