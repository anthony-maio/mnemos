# Native Hermes lifecycle setup verification

Both preserved attempts pass import/configuration preflight. Neither makes a
model request, service request, or memory mutation. Native lifecycle acceptance
remains open.

The committed runner `773738fac688b87306d8667f9cde331e410e309b` produced
[v2/report.json](v2/report.json) using the verified runtime `ea88dd4` wheel,
SHA-256 `cd7896291f51f8e5ddf75c8a56b314113e6c0dacbd54b08b12755ac10df2956c`.
It imports the real isolated Hermes checkout at `14a3463` with the previously
verified cwd-only patch, initializes its MemoryManager and the installed wheel's
provider against a disposable project, and discovers all six Mnemos tools.
The native phase exits successfully in 2.502 seconds. The live host remains
unchanged. This manually registers the real provider with MemoryManager; it
does not establish automatic CLI plugin loading or any model-driven behavior.

The earlier v1 attempt also passed, in 2.793 seconds. Its runner inputs were
not yet committed. Preserve its copied executed files and hashes as that
attempt's evidence; its recorded repository HEAD alone does not identify those
uncommitted inputs. V2 records the committed successor and executes a copied
bootstrap. Eighteen copied artifacts have verified hashes in `manifest.json`.
Disposable configuration, authentication descriptors, and extracted packages
are excluded.

The prospective [protocol](../../../plans/2026-09-19-hermes-native-lifecycle-protocol-v1.md)
requires actual model-issued remember, job status, recall, inspect, correct,
and forget calls across fresh native processes and service restarts. Forgetting
includes authorized removal of retained conversational copies. Separate API
source replay is labeled explicitly. Control facts, source rows, and evidence
must survive unchanged. The runner refuses model execution while the recorded
supplemental queue or any supplied coordinator/worker PID remains active.

Focused verification: 40 tests passed across the trace evaluator, queue/control
guards, and documentation consistency; Black passed for six files and the
configured Python 3.10 mypy passed for four benchmark modules. A final ordering
check restricts job-status evidence to after the remember response; the 19
trace tests passed again. Runtime package counts remain those of `ea88dd4`.
