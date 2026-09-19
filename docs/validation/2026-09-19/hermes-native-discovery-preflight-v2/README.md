# Native Hermes discovery on the repaired wheel

The import/discovery preflight passes on verified runtime `3730fd9`, wheel
SHA-256 `3772398a8e11b92d2cffd285b0571a733a9fd63cbdf932c30519d933cf0a6dd2`.
The native phase took 2.625 seconds and exited zero. Runner snapshot
`bdf56a3` retains the committed protocol and probe inputs.

Hermes `14a3463` uses the same isolated cwd-only patch as the earlier native
run. Its actual plugin loader discovers the installed Mnemos provider, then
initializes it through MemoryManager with the disposable project and shared
Hermes home. All six tools are present. The trace has no model request,
service request, or memory mutation. Full native lifecycle acceptance remains
open; this validates discovery of the new package before model execution.

`artifact-index.json` binds nine exact outputs and executed inputs. Disposable
configuration and authentication descriptors are excluded. The earlier
[discovery preflight](../hermes-native-discovery-preflight-v1/README.md)
retains its original wheel and source bindings.
