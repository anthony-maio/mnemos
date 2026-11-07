# Codex v10: inspection passes, forget preview argument omitted

The inspection alias works in the installed native workflow. Teaching, first
restart recall, same-ID correction and second restart recall all passed, with
12 correlated calls and complete tool inventories. Three native inspections
used `fact_id` successfully. The overall attempt failed during forgetting:
the host twice omitted the required `expected_version` from its preview request.

Product `acc7a317` accepts `fact_id` and the existing `chunk_id` name, rejects
blank or conflicting identifiers, and retains host-resolved scope. All 39
focused MCP checks passed, including 13 new checks and a real stdio session
that compared evidence, blocked foreign-project inspection and verified no
SQLite mutation. Strict type checking and formatting passed.

The new wheel passed metadata checks and the
[installed CLI/MCP restart-recall check](installed-smoke-acc7a317.json) with
real BGE embeddings. The external native environment matched all 95 package
files, 84 committed Python files and 57 expected dependency versions. Seventeen
embedding assets retained their hashes. Wheel SHA-256:
`762c860dc6766f4349211c6ccc12ce69f7553e5a5159cc3775c8a2ae80e994c0`.

The [v10 protocol](../../plans/2026-11-07-native-codex-inspect-v10.md) was
committed at `4cd5775b` before inference. It retained the original lifecycle
requests and v9 inventory requirement. Codex CLI 0.164.0-alpha.11.1, the pinned
Code Mode binary, `gpt-6-astra`, high reasoning effort and 180-second phase
deadlines were unchanged. There was one fresh attempt and no corrective message
or phase retry. The unchanged harness retains its 179 focused passes. The older
3,495-pass full regression remains bound to its original source.

| Phase | Result | Evidence |
| --- | --- | --- |
| Teach | Passed | Three correlated calls; original job completed |
| First restart recall | Passed | Two calls; same fact and evidence at version one |
| Correct | Passed | Five calls; same fact ID updated to version two |
| Second restart recall | Passed | Two calls; corrected content and evidence |
| Forget | Failed | Two preview errors and one failed code cell; later versioned preview and deletion succeeded |
| Final restart recall | Not run | Attempt stopped before replay and final recall |

The initial parser error said the code cell did not match its completed model
call. Inspection of the original trace established that the IDs and source code
did match. The cell was marked failed because it tried to read `result` from
an error response after omitting `expected_version`. Two original MCP responses
identify the missing argument. The later preview with version two and reviewed
deletion succeeded, but they do not erase those failures. The strict rejection
is preserved; the failed phase has no accepted inventory receipt.

An independent audit reconstructed the four passed phases from native model,
cell and MCP IDs, verified their inventories and original prompt text, and
reproduced the failed-phase rejection. Its first audit attempt stopped because
Windows prompt files contain CRLF while the recorded hash covers the exact
native prompt string. Revision two verifies both representations separately,
checks the native message against the original string, and retains both hashes.
The initial audit and its failure log are archived. No inference was repeated
and no native result changed.

After the host's eventual deletion, the target and the scenario's forgotten
labels were absent from the database, and all three retained controls were
unchanged. SQLite integrity and foreign-key checks passed. This is database
evidence only: no exact source replay or final native restart recall occurred.

Cleanup passed. All ten recorded process identities were absent, copied
credentials and runtime descriptors were removed, and normal authentication,
configuration and model assets were unchanged. The private archive contains
732 verified files, including the wheel, original failures, native streams,
focused checks, installation receipts and both audit attempts. SHA-256:
`c7123f23553e145ee50858696877799a62ce06d0741105122ffdbcdc1d4c6012`.

The remaining workflow issue is reliable discovery and use of the preview's
required version. Preserve that guard and the failed result while improving
how the host sees the contract. The pilot remains inactive with zero days;
this result does not establish quality gains, meet the failed hook latency
target or authorize live activation. No publication changed.

- [Sanitized receipt](native-codex-v10-receipt.json)
- [Original inspection failure](native-codex-v9.md)
