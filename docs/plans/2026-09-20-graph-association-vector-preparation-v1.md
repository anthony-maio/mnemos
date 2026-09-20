# Source association vector preparation v1

Status: prospective preparation protocol, NOT RUN. This step prepares vectors;
it does not measure graph reachability, ranks, or quality.

## Inputs fixed before the first request

Use the five authored cases in
`benchmarks/local_memory/datasets/source-association-development-v1.json`
and `benchmarks/local_memory/prepare_source_association_vectors.py`. Commit
both before contacting Ollama. Record the full source revision and both file
hashes, and reject untracked inputs or working bytes that differ from HEAD.
LF/CRLF normalization may be recognized explicitly; preserve the hash of the
actual bytes read. Output must be a fresh directory outside the repository.
Do not replace an existing attempt, including a failed partial attempt.

The cases have exactly 36 query-eligible facts each. The orchard case adds
four foreign-scope controls and includes a permitted project/workspace
relation among its primary facts. No query, content, target, or case may be
changed after seeing vectors or subsequent graph observations. All cases
remain in the experiment even if a target is a direct retrieval candidate.

## Local model and text contract

Use only `http://127.0.0.1:11434`, following the user's replacement of the
failed .35 machine with local Ollama on the RTX 4080. The model is
`nomic-embed-text:latest`, with digest
`0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`.
Require the exact digest in `/api/tags` before and after preparation. Record
`/api/version`, `/api/show`, and `/api/ps`; reject a change in the selected
model's effective metadata. Do not infer GPU execution solely from installed
hardware. Record the observed loaded model state.

Query, document, and symmetric prefixes are all empty strings. Embed each
unique exact text once and reuse its vector across its recorded roles. This
matches the current unprefixed production default and the earlier frozen
runs. It is not a test of Nomic task prefixes. Source inspection found only a
two-vector prefixed smoke check, not a prefixed quality comparison. That gap
remains open; it does not establish why earlier quality results were poor.

Include query text, every primary fact, every source event, foreign controls,
and correction replacement content. Preserve explicit role bindings even
when identical text is deduplicated. Do not infer or synthesize missing
vectors. Use deterministic batches of 16, a 60-second timeout per request,
`/api/embed` with `truncate: false`, and no application retries or fallback.
Each response must contain exactly one finite, nonzero, 768-dimensional
vector per input. Record the actual request parameters; omitted Ollama
options retain the recorded model/server defaults.

## Evidence and stopping conditions

Save attempted request bytes before dispatch and exact response bytes before
parsing, with lengths, hashes, timestamps, URL, method, and HTTP status. Keep
completed batches and the attempted request if a later call fails. Record a
failed artifact on transport, status, parse, vector, digest, metadata, or
input-drift failure. Recheck source revision and input bytes after all calls.
Successful preparation means the complete vector artifact passed these
checks. It says nothing about associative recall or model quality.

No similarity calculation, target ranking, arm comparison, or case selection
belongs in this step. The separate graph measurement protocol must bind the
completed vector artifact digest and the committed measurement implementation
before running its five cases, three arms, and three repetitions. A failed
preparation is retained and diagnosed before another uniquely named attempt.
