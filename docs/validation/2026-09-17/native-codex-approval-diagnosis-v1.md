# Native Codex approval diagnosis

The preserved [retrieve-v2 stream](native-codex-retrieve-v2/native-stdout.jsonl)
records one tool call followed by `user cancelled MCP tool call`, without a
service result. Its [runner](native-codex-retrieve-v2/runner.py.txt) invokes
noninteractive `codex exec --json --sandbox read-only` with closed stdin.
The tested CLI was 0.146.0. Mnemos retrieval declares `readOnlyHint=false`
because retrieval updates durable access metadata.

An approval-path cancellation is a plausible explanation. An
[upstream report for Codex 0.139.0](https://github.com/openai/codex/issues/29857)
describes the same cancellation in `exec` while interactive Codex succeeds.
That report is supporting precedent, not proof of the cause in this version.
The Mnemos run did not retain an approval event or server-side arrival trace,
so transport and timeout explanations remain open.

The next discriminating check is a fresh isolated interactive CLI session
using the generated MCP configuration, the same sandbox, and the normal
approval flow for one explicitly authorized fixture retrieval. Record the
approval, actual MCP result, and service-side arrival together. Preserve
the truthful tool annotation and normal sandbox restrictions. A fabricated
result or a global approval bypass would not establish this contract.

The subsequent [interactive check](native-codex-interactive-v1/README.md)
displayed a normal per-call approval and returned an actual MCP validation
error after approval. Codex supplied `allowed_scopes` as a list; the installed
tool accepted only a string. The service received no retrieval dispatch.
This establishes a separate argument compatibility failure, not successful
retrieval or the sole cause of the earlier cancellation. The earlier run
and the interactive check also used different source revisions.

The subsequent [interactive v2 check](native-codex-interactive-v2/README.md)
used installed source `30eaf42`, including support for structured scope lists.
Normal per-call approval was followed by a successful native MCP result and
a matching service dispatch. The result contained the seeded project fact
and excluded the foreign fact. The process later timed out during shutdown.
This establishes bounded retrieval after the argument repair. Changes in
source, fixture location, and invocation prevent attributing the earlier
noninteractive cancellations solely to approval handling.
