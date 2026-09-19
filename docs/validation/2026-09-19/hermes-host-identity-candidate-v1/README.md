# Proposed Hermes durable message identity compatibility patch

The [proposed patch](../../../integrations/hermes-durable-message-identity.patch)
targets Hermes `14a346345486418aa85ed6526549412ee2665ade`. It preserves
durable row IDs when late CLI resume and compression recovery reload stored
messages, and adopts durable IDs onto the retained warm prefix during CLI
undo/retry. It changes six call sites across three host files. It is applied
only in `/tmp/mnemos-hermes-durable-identity-v1`; the live installation and the
cwd-only checkout reserved for the native lifecycle run are unchanged.

Late resume with the actual SessionDB fails the new ID assertion on the
unpatched source and passes on the candidate. Two stronger undo/retry tests
start with IDs removed from warm messages, call the actual CLI methods, and
inspect the history those methods publish. Both fail on unpatched source and
pass on the candidate: retained IDs are `[1, 2]`, and richer warm fields
survive. Compression-adoption tests call the actual helpers with loader
doubles and verify the requested ID-preserving contract. They do not prove
model-driven compression or continuity across cloned rows.

The affected upstream suite initially has one failure: it explicitly expects
CLI undo to omit `_row_id`. The proposed test change now requires those IDs;
other upstream checks remain intact. It is an intentional metadata contract
change, not a waived failure. The final affected set passes 28 cases. An
independent run reproduced all 28 passes in 8.36 seconds and verified that
imported host modules originate in the candidate checkout. This is functional
test timing, not a performance measurement.

Earlier attempts remain in `preceding-evidence`. The first report's
`candidate_verified` label is too strong: its fixture only called an existing
SessionDB option directly. The second adds call-contract checks but its undo
test still passes without the patch, and its affected suite fails the old
CLI metadata expectation. Only the v3 tests exercise the changed undo/retry
call sites with ID-less warm histories. Original failures and reports are
preserved, including their limited or incorrect labels and shorthand paths.

`report.json` records the independent commands, source revision, imported
origins, patch binding, and exit codes. `evidence-index.json` binds its raw
outputs and preceding evidence. `executed-tests/index.json` separately binds
the four exact test files read from the unchanged candidate after that run;
their normalized bytes match the exported focused tests and test patch.

This is a reviewable upstream compatibility candidate, not an installed host
upgrade or a new Mnemos wheel. Native model lifecycle, cross-host continuity,
compaction clone provenance, and migration of existing capture identities
remain open. No model, Mnemos service, live memory, or live configuration was
used by these tests.
