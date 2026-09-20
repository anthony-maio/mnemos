# Native Claude embedding outage v1: failed verifier comparison

The frozen attempt on product `4bbab4f`, runner `897247b`, failed on September
20 at the comparison between the native MCP result and service response.
The MCP tool adds `schema_version=1` to each fact. V1 required exact equality
before that existing transport projection and stopped before its final state
gate. Its verdict remains failed; this is not a passed native acceptance run.

The recorded conversation contains one actual retrieval tool call and a later
answer with the stored marker. The response reports degraded lexical retrieval,
null semantic scoring, and the authorized project fact. Offline archive checks
confirm those bounded observations. They do not retroactively satisfy v1's
frozen equality gate. Cleanup passed and no embedding response occurred.

The provider targeted an owned non-listening loopback socket. Vectors were
synthetic 768-dimensional fixtures, all five cognition modules were disabled,
and background processing was stopped. No active Ollama evaluation was used.

The [successor](../native-claude-embedding-outage-v2/README.md) accounts only for
the existing MCP schema marker, with a test invoking the actual MCP tool. All
substantive retrieval and cleanup gates remain. The product is unchanged.

Run `python verify.py` here for offline artifact hashes and bounded native
checks. The transcript projection omits hidden reasoning and preserves message,
tool, model, and session evidence, with hashes of the private full source.
Credentials and the database are not included. The README is outside the
evidence index. Initial verifier regressions and passing focused checks are
retained separately from the native result.
