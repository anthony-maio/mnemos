# Native Hermes plugin discovery preflight

The successor preflight passes using runner `f10bf6c`, verified runtime
`ea88dd4`, and the same isolated Hermes `14a3463` checkout with its cwd-only
patch. The native phase takes 2.434 seconds and exits zero. No model request,
service request, or memory mutation occurs; full native lifecycle acceptance
remains open.

This uses Hermes's actual `plugins.memory.load_memory_provider("mnemos")`
entry-point discovery, then its MemoryManager to initialize the discovered
provider. The trace verifies the disposable project, shared Hermes home, and
all six tools. It supersedes the narrower manual-registration preflight while
preserving [those earlier attempts](../hermes-native-lifecycle-preflight-v1/README.md).
It still does not establish complete CLI startup or model-driven behavior.

Source review also found that the prospective API replay helper had omitted
the Hermes home from initialization. The native model process and replay must
share that home because it contains the durable occurrence-time ledger. The
successor supplies and verifies it. No failed model run was hidden or retried;
model execution has not begun. Existing executed helper copies remain unchanged.

Forty focused trace, queue/control, and documentation tests pass; configured
Python 3.10 mypy passes all four benchmark modules. The runtime wheel is
unchanged, with SHA-256
`cd7896291f51f8e5ddf75c8a56b314113e6c0dacbd54b08b12755ac10df2956c`.
Nine raw artifacts and executed inputs are bound by `artifact-index.json`.
The archive excludes disposable configuration and authentication descriptors.
