# Complete hook startup diagnostics

The [installed performance run](installed-performance-v3.md) missed the 100 ms
hook requirement. Separate diagnostics on that same `749f1213` wheel show a
substantial cost before a hook dispatches any event.

| Fresh subprocess | Samples | Median ms |
| --- | ---: | ---: |
| Interpreter only | 7 | 79.85 |
| Import settings | 7 | 416.55 |
| Import client | 7 | 610.72 |
| Launcher configuration probe | 7 | 464.63 |
| Launcher and empty hook | 7 | 669.54 |

These cases ran in interleaved rounds without inference or service requests.
The empty hook received `{}` and exited without dispatch. No database or
runtime was created. Seven additional complete Stop captures ran in a private
copy of V3's final database, with a median of 985.71 ms. An eighth capture ran
under cProfile. All eight events and their durable jobs were verified after
service shutdown, and the original V3 database stayed unchanged.

The capture profile records 326.14 ms importing settings, including 107.04 ms
importing the retrieval domain types. Actual settings loading takes 27.21 ms,
project identity resolution 66.71 ms, and initial HTTP client construction
92.17 ms. Twelve private-path ACL checks take 8.92 ms cumulatively. Imports and
file-system operations dominate this sample. These profiled durations overlap
and include instrumentation overhead; they are diagnostic observations, not
additive parts of an accepted latency score or a replacement for V3.

One unnecessary dependency is now separated: settings and capture safety
configuration imported `RetrievalConfig` through the scoring module, which
also constructed memory and cognitive-state models. The configuration class
moves to its own small module. Its fields, defaults and validators stay the
same, and `mnemos.retrieval.RetrievalConfig` continues to expose the same class.
Both complete settings and retrieval JSON schemas match the installed baseline.

Three new import-boundary checks fail before this change and pass afterward.
They reject retrieval domain and inference imports during settings, client and
Codex adapter startup. The existing provider-import and public-export checks
are retained. There are 37 passing configuration/retrieval checks and 68 passing
adapter, client, service and import checks, with three shared cases. Black and
strict mypy pass for the changed source. This establishes the dependency repair;
it does not yet measure a speedup or meet the complete hook requirement.

The diagnostics and original inputs are preserved in a verified 126-member
private archive, SHA-256
`90962fbf8936ab1956bed1873be2e7649331ba5136f773ee045aac6cb2d42986`.
The diagnostic service exited gracefully and removed its descriptor. Normal
maintenance was enabled during real captures; its embedding count was not
instrumented. These runs add no native-host acceptance, quality result or pilot
days.

- [Diagnostic receipt](hook-startup-diagnostics-v1-receipt.json)
- [Original complete-command result](installed-performance-v3.md)
