# Native Claude pilot integration v2

Apply the v1 protocol with the following runner corrections, frozen before
one fresh attempt. Preserve the v1 failure and its original cleanup result.

V1 completed its explicit controller seed, then failed before native submission.
The seed service PID was absent after the Windows MCP stdio client closed, but
its descriptor remained. The later service-stop command could not reach it.
The installed MCP client uses Windows process-tree termination. Stop the seed
service while the MCP context remains open, then close that context. This is a
controller correction, not evidence of native restart or recovery behavior.

Validate the installed inspection schema and assert the native call's actual
`chunk_id` argument. The v1 runner incorrectly checked `fact_id`, which is the
internal service payload name. No native call reached that assertion in v1.

Record shutdown stdout, stderr and return code. Preserve independent before
and after hashes for both source credential files and the inactive pilot files.
A shutdown failure must not be reported as proof those files changed.

Keep the same product, wheel, executable, model, prompt, fact, two-tool limit,
180-second and $0.50 bound, physical capture checks and privacy rules. Use a
fresh disposable root. No model retry, live host edit or pilot day is authorized
by this protocol. This attempt tests the amended controller and the original
draft launcher and hooks; it does not repair service lifecycle behavior.
