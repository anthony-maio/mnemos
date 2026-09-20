# Hermes candidate hybrid-copy audit

The actual host compressor can combine a generated summary and an original
user message in one model-facing carrier. The candidate projects those into
separate durable evidence components. Copying an unchanged hybrid through
the model-facing restore path still changes the summary's canonical identity.

The valid v2 probe calls `ContextCompressor._merge_summary_into_tail_row`,
publishes a compression child, reloads with
`get_messages_as_conversation(include_row_ids=True, repair_alternation=True)`,
and publishes a grandchild. The original user retains canonical row 1, but
the unchanged derived summary moves from canonical row 2 to row 3. A copy
must preserve both identities. Recomputed summary text is a separate case
that needs its own new derived identity and verification.

The second v2 case alters a copied summary component's role and content in
the isolated SQLite fixture. The candidate rejects it with `ValueError`.
This supports that bounded tamper check; it does not close compaction.

The earlier v1 probe supplied an unsupported public-loader keyword and did
not exercise the copy case. Its error, second-case result, source patch, and
runner remain preserved. V2 corrects the loader invocation. Both runs record
identical before/after patch hashes, and their patches bind the candidate
state. This is fixture evidence, not model inference or a live installation.

All eight indexed artifacts match the copied bytes. This README is outside
the evidence index.
