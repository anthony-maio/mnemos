# Native Claude startup-wait control v16

The installed `56182d9` wheel was exercised through Claude Code 2.1.191 with
`MCP_CONNECTION_NONBLOCKING=0` only in the disposable child environment. The
connection timeout, default permissions, tool allowlist, 60-second native
deadline, and model budget were retained. No live host settings were changed.

The initial catalog reported Mnemos connected with all 14 tools. Claude made
correlated correct, consolidate, and retrieve calls. All six native correction
checks passed, including stable target identity, version 2, exact replacement
content, and absence of the old target version in the returned facts.

The complete probe failed afterward: direct retrieval after maintenance did
not return the target. The version and content checks immediately before that
retrieval passed. Restart and native forget were not reached. The original
failure remains failed; the cause is not inferred from the native correction
success. Hook capture also retained degraded diagnostics.

This is a successful initial catalog and native correction observation, not
full lifecycle, extraction-quality, or general startup-reliability acceptance.
The fixture uses constant unit-vector embeddings and disabled cognition.
Raw transcripts remain in the private fixture; the archive records their
location and hashes. The private auth copy and service descriptor were removed.
