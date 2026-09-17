# Native Codex interactive retrieve v1 checkpoint

This archive preserves one bounded interactive Codex 0.146.0 run against the
installed Mnemos wheel bound to source commit `a607a5a6b0a8c2d4f0c4ed00e3b21167cf96f854`.
The wheel SHA256 is `4ffcc30358ed95d885f8e86368510bb9de70ec8d3933f75eb12097a3315738ce`.

The command used normal interactive approval with `--ask-for-approval
on-request`, `--sandbox read-only`, and inherited terminal input/output.
The generated fixture hooks were reviewed and trusted through the normal UI;
no approval, sandbox, or hook-trust bypass flag was used. The operator
saw the per-call Allow prompt and allowed the explicitly requested Mnemos
call. This is a real native host interaction, but the run is negative.

Codex dispatched an `exec` custom tool call that invoked `mnemos_retrieve`.
The paired result was an MCP error: the request supplied `allowed_scopes` as a
list while the installed Mnemos tool schema expects a string. No successful
retrieval result was returned. The service dispatch trace was empty and both
fact access counters stayed unchanged. The interactive process then reached
the 180 second deadline and was cleaned up. After the tool result, the same
process displayed fresh-profile Windows sandbox setup. The operator selected
the non-admin option; setup was still displayed at the deadline. This terminal
observation is separate from the preserved MCP error. Selected terminal-tool
outputs and operator actions are in observed-terminal-chunks.json; several
outputs were truncated by the terminal tool, so this is not a full recording.

Only the matching custom tool call and paired output JSONL records are retained
in `selected-mnemos-events.jsonl`. The full Codex session, prompts, reasoning,
settings, authentication data, databases, and plugin cache are excluded.
The report retains the original sanitized run metadata and cleanup proof.

This archive proves normal approval interaction and a real tool validation
failure. It does not prove a successful Mnemos retrieval, isolate the cause of
the earlier noninteractive cancellation, or provide model-quality evidence.
