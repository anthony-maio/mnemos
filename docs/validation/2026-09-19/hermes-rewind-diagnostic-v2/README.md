# Hermes rewind identity diagnostic v2

This diagnostic keeps the outbox pending so the result measures identity persistence instead of a local client acknowledging every capture. It uses four messages from two turns, an exact duplicate, a same-session `rewound=True` replacement, a changed final assistant message, and the same transcript under a new session ID.

The run used the verified `ea88dd48e503adee811a524bf76923e153c17384` wheel with SHA256 `cd7896291f51f8e5ddf75c8a56b314113e6c0dacbd54b08b12755ac10df2956c` and Hermes revision `14a346345486418aa85ed6526549412ee2665ade`. The module origins identify the actual isolated Hermes MemoryManager and the wheel adapter. Runtime files were created under a disposable temporary directory outside the repository and removed after collection. No Mnemos service, model, GPU, live configuration, or native CLI ran.

The initial four messages caused the provider to generate four pending payload IDs. Repeating the same transcript made four additional capture requests while the durable outbox stayed at four entries. Repeating the transcript after the same-session rewind did the same thing: four new requests, the same four pending IDs, and the same occurrence timestamps. The expected logical count after a genuine rewind and regenerated last turn would be six observations, so identity acceptance is false and the collision is reproduced.

Changing only the final assistant message added exactly one pending ID, leaving the surviving prefix IDs unchanged. Switching to a new session and submitting the original four-message transcript added four new IDs. The final pending outbox contained nine unique payload IDs: the original four, one changed suffix, and four new-session observations.

This confirms the boundary problem without pretending that queued local records were committed by a service. It also keeps exact duplicate replay idempotent. A repair still needs a host-supplied rewind generation, retained boundary, removed turn ordinal, or native turn identity. Re-keying every surviving prefix when `rewound=True` would create false new observations for messages that survived the rewind.

The earlier v1 attempts are preserved in `original-attempts`. The first attempt
had zero captures because its provider lacked a registry, despite its report
incorrectly saying `passed`. Its stderr records the initialization failure.
The v1 rerun used a local client that acknowledged and removed every record,
so its identity observation remains useful but its pending-outbox evidence is
limited. Neither earlier report is promoted to lifecycle acceptance.

Independent review checks the raw phase counts, unchanged IDs/timestamps on
duplicate and rewind callbacks, one added changed-suffix ID, and four added
new-session IDs. The archived index binds the original attempts as well as the
successor. The native `/undo` implementation supplies `rewound=True` but no
retained transcript boundary at `hermes_cli/cli_session_mixin.py:813` in the
pinned host. This probe calls that MemoryManager boundary directly; it does
not invoke the interactive command or claim service persistence after a crash.

This is installed API callback evidence, not full native CLI acceptance.
