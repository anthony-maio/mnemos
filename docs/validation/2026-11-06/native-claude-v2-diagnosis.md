# Claude v2 forgetting diagnosis

The first forgetting call included two target-linked source IDs in
`selected_source_event_ids`. Those IDs appeared under the preview's
`target.source_event_ids` and `target.affected_sources`. They describe cleanup
already included with the target; they were absent from the top-level list of
additional selectable source events. The service returned
`related_forget_selection_invalid` with `committed=false`.

The model then removed exactly those two IDs and repeated the call with every
other argument unchanged. All remaining selected sources were eligible in the
same preview, and the second call committed. The strict native parser correctly
rejects the earlier unsuccessful call. The original attempt remains failed;
this diagnosis does not change its acceptance rule or resume its final phase.

The retained trace places the first request/result at lines 560/561 and the
second at 694/695. Its SHA-256 is
`caf5bc492c3c2a6d922a60a19dc6a82cf9e67d451e3865133226150c5bbc7407`.
A read-only scan of the final database found neither synthetic forgotten label
in facts, source events, evidence, revisions or job payloads. This confirms
the narrower observed deletion result, not restart/replay acceptance, which
was never attempted. The [diagnosis receipt](native-claude-v2-diagnosis.json)
records the exact argument-set comparison and database counts.

The tool descriptions previously referred generally to IDs from a matching
preview. They now identify the exact additional-selection fields and explain
that target-linked cleanup is already included. Parameter descriptions expose
the distinction directly in the MCP schema. The operating guides and transport
contract give the same instructions. Storage authorization, preview freshness,
dependency checks and deletion behavior remain unchanged.

Forty-nine existing transport, deletion-safety, rejection, precondition and
native-parser tests passed. The live FastMCP catalog exposes the new parameter
descriptions, and formatting and strict types passed for the changed module.
An installed artifact and a new native attempt must establish whether the
clarified interface prevents this mistake in actual use.
