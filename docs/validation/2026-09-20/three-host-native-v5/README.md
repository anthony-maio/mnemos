# Three-host v5: worker starts, Hermes scope arguments rejected

The single scheduled native attempt ran on September 20, 2026, from 08:26:24
to 08:28:53 UTC. Codex and Claude teaching passed. Hermes recall failed
because its first two retrieval calls supplied unsupported `current_scope`
and `allowed_scopes` arguments. Both calls were rejected. Query-only retries
returned both asserted facts at version 1, but do not erase the original
failures. Correction and all later phases did not run.

## Worker and instruction evidence

Both service starts reported `maintenance.scheduler_running=true` under the
recorded 0.25-second/zero-idle benchmark schedule. The worker repair therefore
has startup evidence in a real native run, in addition to its offline
correction/restart regression. This run never reached native correction and
does not establish that a native model observed a completed correction job.

The generated project `AGENTS.md` explicitly tells agents to use
`current_scope=project` and `allowed_scopes=project`. That exact instruction
appears in the system message of all three archived Hermes provider requests.
The Hermes retrieval schema forbids both parameters, and the first two tool
calls supplied those exact fields. This establishes contradictory generated
guidance, not a controlled estimate of its causal effect. The scope firewall
correctly rejected unsupported arguments and must remain unchanged.

## Binding and limits

- Protocol: [v5](../../../plans/2026-09-20-three-host-continuity-protocol-v5.md).
- Runner: `3c3849f`; full revision and input hashes are in each report.
- Product source: `f2f741443e60f838b392ae8ea1d1a0b4d8c6c1f2`.
- Wheel SHA-256: `8552345d2c17c4c037347961eb3043d13b20f790f54a9ccc86a096add6197b0f`.
- Codex requested and observed `gpt-5.5`; native prompt and call correlation
  checks passed. It exited normally in 97.786 seconds.
- Claude exited normally in 16.945 seconds; Hermes in 24.015 seconds.
- Three Hermes responses have verified loaded-profile records. The local
  Qwen alias and real Nomic embeddings remain pinned; all five modules were off.
- Separate import preflight passed. Phase and parent cleanup passed, with
  copied cloud authentication and runtime descriptor removed.
- Preparation passed 167 offline cases, including 11 documentation checks.
  Product package test totals remain those of the earlier frozen checkpoint.

The controller trusted the disposable project and generated hooks, approved
the requested Codex store and consolidation, and sent two Ctrl+C characters
after the final answer and Stop hook completed. There was exactly one native
user prompt. No controller message or model correction was added.

The [evidence index](evidence-index.json) binds 137 files. It includes reports,
the [generated instructions](native/project/AGENTS.md), service log, final
state, service dispatches, selected native cloud records, Hermes request and
response traces, and frozen inputs. Cloud source mappings preserve original
line hashes; full cloud sessions, credentials, and runtime tokens are omitted.
The README and index itself are outside the index.

The [v4 partial result](../three-host-native-v4/README.md) remains distinct.
Neither run completes three-host continuity. The next investigation is the
shared instruction generator; do not relax the adapter's scope authority or
change native acceptance criteria to hide incompatible instructions.
