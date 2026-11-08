# Canonical delivery artifact validation

The wheel built from canonical commit `c77933402a8bb4629211f24be22c3555fc872a76`
passed an isolated consumer installation and the shipped CLI/MCP restart-recall
example. Version remains `0.7.0b1`. This candidate includes the IPv6 dispatch
repair; the original pilot remains bound to its earlier wheel.

The build produced a source archive, then built the wheel from that archive
using isolated Hatchling 1.33.0. The first no-isolation attempt failed because
Hatchling was unavailable; it produced no artifact. The successful build did
not install a backend into the existing validation environment.

Archive verification compared 96 wheel package/resource files and 103 source
archive files against the canonical checkout. Every wheel RECORD digest and
size passed, every source Python module was present, and the packaged client
matched the repaired source. The receipt identifies both artifact hashes.

The consumer installation used Python 3.12.11 outside the source checkout.
Its 57 resolved distributions and metadata hashes were retained privately.
Checks verified installed module origin, package/runtime version agreement,
console and Hermes registration entry points, the type marker, three bundled
benchmark datasets, and disabled experimental module defaults.

The example used real FastEmbed BGE small embeddings with a fresh database,
project registry and runtime configuration. Explicit remember completed before
service shutdown. A new service process returned the same fact ID and source
evidence through the CLI. MCP stdio initialization, tool discovery, recall and
inspection passed; a separate project returned no facts. Service shutdown and
disposable workspace cleanup completed. The smoke process exited zero.

This is a local package and persistence result. It does not rerun the full test
suite, test a live IPv6 listener, establish native-host acceptance, or satisfy
semantic quality, complete-command latency or daily-use gates. The smoke helper
removes its temporary database and service log; its aggregate receipt and
resolved dependency metadata are retained. No tag, push, publication, pilot
activation or normal host configuration change occurred.

See [the artifact receipt](canonical-artifact-v1-receipt.json) for exact hashes
and individual integration checks.
