# Hermes candidate component-write audit

An isolated, uninstalled Hermes compatibility candidate at base
`14a346345486418aa85ed6526549412ee2665ade` exposes four regressions in the
actual SessionDB write/projection path. The exact candidate patch is retained;
these findings do not describe the released Hermes code or a live upgrade.

- A single `append_message` user write persists successfully, but the new
  `get_memory_evidence` projection returns an empty list.
- A batch tool response raises `ValueError` during evidence-component insert.
- An assistant tool call with null content raises the same error.
- A multimodal user message with a text part raises the same error.

The latter three errors read `memory evidence component has invalid role or
content`. The memory integration must preserve valid host transcript writes
and project only capturable evidence. Single-message writes need the same
durable provenance support as batch writes. This probe records outcomes;
it does not assert rollback behavior or cover every host content shape.

The database fixtures remain under the named WSL temporary directory. No
model, live configuration, or live memory store was used. This is preserved
negative development evidence. A successor must rerun the affected paths and
the host suites before making an integration acceptance claim.

The evidence index binds five raw/helper artifacts. This README is outside
the index.
