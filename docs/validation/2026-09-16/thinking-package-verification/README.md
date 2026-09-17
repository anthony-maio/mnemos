# Ollama thinking package checkpoint

Frozen source `d94ded67a6b3fe6b6f7474a74518d4efb143698e` passed 811 tests with four
platform-specific skips, Black, compatible strict mypy, JavaScript syntax,
build, and Twine. A fresh MCP-enabled installation passed source/asset parity
and the instrumented Claude adapter restart/replay check. This checkpoint
does not add native-host conversation evidence.

Wheel SHA256: `af0ba55d1b39498133bf9cd0e022a00850291e4f778edd77213d51ee4b54d008`. Distribution binaries and temporary
fixtures are not included in this archive. The reports bind their hashes and
the exact environment commands. No live configuration or memory data changed.

The evaluation coordinator is preserved before its real-model execution.
It pins source, model digests, thinking mode, and verification reports. The
offline v2 check exercises lifecycle attempt selection with synthetic reports;
its source inspection checks the command rewrite. This is not a full queue
execution or a fault-injected process restart test.

An earlier offline v1 report was manually rehashed after its helper changed,
so it does not bind the executed helper and is excluded from accepted evidence.
The original remains in ignored scratch storage. The v2 helper was actually
executed into a new output directory; its report and both helper hashes were
verified before archival.

Real-model extraction quality, held-out module benefit, latency gates, complete
native-host continuity, and pilot acceptance remain separate measurements.
