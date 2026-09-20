# Native Claude embedding outage v2

## Purpose

This protocol checks the bounded retrieval behavior exposed to one native Claude
Sonnet 4.6 conversation when the local embedding endpoint is unavailable. It
uses a disposable SQLite store and one real Claude retrieval tool dispatch.

The experiment does not measure embedding quality, model quality, recall lift,
or general host reliability. The seeded vectors are synthetic 768-dimensional
schema fixtures. They make the database valid for the configured embedding
dimension while the retrieval request is forced through FTS lexical fallback.

## Frozen run

The runner requires the exact source and installed-package verification records
for product commit `4bbab4f01fd6b5a8e1e4d76504eaa8d59d349da8`. It refuses a
missing, changed, or dirty input and refuses an existing output directory.
The output is exactly:

`C:/Users/Ant/AppData/Local/Temp/mnemos-native-claude-embedding-outage-v2`

The runner uses the verified Claude plugin and `claude-sonnet-4-6`, disables all
five cognitive modules and scheduler processing, and permits only
`mnemos_retrieve`. It does not start a provider, Ollama model, or background
worker. It keeps an owned `127.0.0.1` TCP socket bound without calling
`listen()` for the full attempt. The Ollama embedding provider targets that
non-listening port, so the port cannot be claimed by another process and no
embedding response can be mistaken for a real provider result.

The target fact is `The deployment marker is maple-heron-719.`. A separate
foreign-project fact is seeded as a scope control. The native prompt asks for
the deployment marker, explicitly requests one `mnemos_retrieve` call with the
query `deployment marker`, and does not contain the answer text. The native
transcript tool result and the recorded service dispatch must both show the
target result.

## Acceptance

The run is accepted only when all of these records agree:

1. Source and installed checks pass with the pinned commit and wheel bytes.
2. Health reports private SQLite storage, all five modules disabled, and no
   scheduler or background maintenance.
3. The native Claude process completes with the approved retrieval tool only.
   The launched session and model must match every native conversation record.
   Require one successful tool result, correlated by tool-use ID, and a later
   final answer containing the stored marker. Reject extra prompts, other
   tools, duplicate results, absent model identities, and cross-session records.
4. The retrieval response is `degraded`, explicitly reports
   `retrieval.mode=lexical_only` and `reason=embedding_unavailable`, contains
   the target in project scope, and exposes `semantic=null`,
   `state_status=not_measured`, `graph_status=not_used`, and `total=null` for
   returned facts.
5. The foreign fact is absent from both returned facts and rendered context.
   Decode the native MCP result and require equality with the single service
   response for query `deployment marker` after applying only the existing MCP
   addition of `schema_version=1` to each fact. Every other field must match. Hook retrievals use the full prompt
   and are recorded separately.
6. SQLite integrity remains valid and the seeded target and foreign content,
   versions, scope identities, and assertion status remain unchanged.
7. No embedding response is observed, credentials are removed, the runtime
   descriptor is gone, and the service shuts down cleanly.
   Cleanup failure must persist a failed report, even if retrieval succeeded.

The result remains negative or partial if the native transcript succeeds while
the service dispatch, scope control, cleanup, or provider-outage evidence does
not. No result promotes lexical fallback to semantic retrieval quality.

## Evidence and limits

The report, dispatch records, health response, native transcript hash, seed
snapshot, package hashes, and cleanup checks are written inside the disposable
output only after the protocol preflight passes. A failed attempt preserves its
report for diagnosis. The experiment is one native conversation with fixed
synthetic vectors, so it cannot establish production outage rates, model answer
quality, comprehensive foreign-scope testing, or multi-request durability.

## Change from v1

V1 remains failed. Its native result comparison stopped at the existing
`mnemos.mcp_server.mnemos_retrieve` per-fact schema marker, before reaching
its post-retrieval state gate. The recorded native answer and service response
both contained the target and degraded status; cleanup passed. This successor
changes only transport-shape comparison and preserves all substantive gates.
A deterministic test invokes the actual MCP tool function to verify the
projection, and altered content, version, scope, or status still fails.
Product, model, prompt, synthetic vector policy, and resource bounds are
unchanged. This is a fresh attempt, not a revised verdict for v1.
