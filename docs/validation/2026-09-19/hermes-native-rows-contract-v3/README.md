# Native Hermes row identity contract

This callback fixture passes on frozen Mnemos source `3730fd9` and actual
Hermes `14a3463` source with the previously isolated cwd patch. It uses the
real SessionDB, post-commit persistence helper, normal resume loader,
MemoryManager, Mnemos provider, and local Outbox. It does not run a model,
interactive CLI, Mnemos service, or an installed Mnemos wheel. Live source,
configuration, and data are unchanged.

The host persists two turns and stamps native row IDs `[1, 2, 3, 4]` onto the
live message dictionaries. Its actual `rewind_user_turn` preserves `[1, 2]`.
Persisting an identical regenerated suffix creates `[5, 6]`, and the normal
resume loader returns `[1, 2, 5, 6]`. The actual provider submits 20 capture requests
across initial capture, checkpoints, regeneration, and provider recreation.
Exactly six queued occurrences remain, with six distinct source references.
No native row IDs are fabricated by the fixture. The local client returns
`queued` and leaves records in the actual outbox.

The earlier [negative diagnostic](../hermes-rewind-diagnostic-v2/README.md)
exercises a transcript without native IDs; its collision remains preserved.
The repair uses available durable IDs instead of inferring new occurrences
from repeated text or rekeying a surviving prefix. It does not establish
identity continuity through loaders that omit IDs, compaction that clones
rows, or migration from previously captured prefix identities.

The two preceding native-row probe attempts ended with `runtime_failed` and
no usable result. Their reports and exact executed helper/bootstrap copies
are preserved in `original-attempts`. Their generated READMEs incorrectly
described successful verification and are not accepted as evidence. Those
helpers also omitted stderr and removed their runtime, so their exception
details cannot be independently reconstructed from the saved reports. The
successor preserves both streams and its disposable runtime.

Hermes detected the Windows-mounted database directory and selected DELETE
journaling. Its exact diagnostic is in `stderr.log`. This is not a WAL,
hard-kill, process-restart, or throughput test. Provider recreation is the
restart boundary exercised here.

`report.json` binds the source revision, actual imported module origins,
adapter hash, helper hash, and bootstrap hash. `artifact-index.json` binds
the raw outputs and preceding failures. The dependency wheel used to supply
the host's import metadata is superseded by the explicitly verified frozen
source path for every Mnemos import; the result's module origins prove that
selection.
