# Native Claude embedding outage v2: lexical recall observed

One native Claude Code 2.1.191 / Sonnet 4.6 conversation passed the frozen
embedding-outage experiment on product `4bbab4f`, runner `c58c5d4`, from
16:10:20 to 16:10:38 UTC on September 20, 2026.

Claude called the actual retrieval tool once with `deployment marker`, received
the existing project fact and explicit degraded lexical status, then answered
with the marker that was absent from its prompt. The native MCP payload matches
the service response after the existing per-fact `schema_version=1` addition.
Every other field matches. The foreign-project control was not returned.
Semantic and total scores remain null; graph and affective scoring are unused.

Independent read-only SQLite checks found both seeded facts unchanged in
content, version, scope, and provenance, with valid integrity. The owned service
stopped, its descriptor and copied credentials were removed, and no embedding
response occurred. Hook retrievals are preserved separately from the native
tool call in the dispatch trace.

The embedding provider targeted an owned non-listening loopback socket. Its
768-dimensional stored vectors were explicitly synthetic schema fixtures;
lexical ranking does not use them. All five cognition modules were disabled and
background processing was stopped. This proves one native lexical retrieval
during provider unavailability. It does not prove semantic quality, extraction
or provider recovery, a latency guarantee, comprehensive resilience, or daily
use readiness. The factory injects the provider into a real loopback service;
default settings-based provider construction is not this experiment's subject.

The [v1 failure](../native-claude-embedding-outage-v1/README.md) remains failed.
The successor changed the verifier's transport-shape comparison, with a test
using the actual MCP tool implementation, and retained every substantive gate.
There were 51 focused checks before this attempt. The underlying
[package checkpoint](../retrieval-provider-outage-checkpoint-v1/README.md)
passes 1,894 source tests and 722 installed cases, four skips each, six
source/package checks, and seven replay assertions.

Run `python verify.py` here to verify the indexed artifacts, native tool/result
correlation, scoped degraded response, final seed state, and cleanup evidence.
The projection omits hidden reasoning while preserving native conversation
evidence; full-source and line hashes are retained. Credentials and the database
are excluded. This README is outside the evidence index.
