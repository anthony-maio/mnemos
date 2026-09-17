# Installed continuity diagnostics

Both runs used the original installed `312b8d6` wheel and frozen two-case test with its five-second client and ten-second service dispatch timeouts. The model providers in these fixtures are synthetic. No Ollama calls were made.

## Valid timeout reproduction (v2)

The fixture was explicitly outside Git. One case failed and one passed in 135.34 seconds. The Codex teaching case returned degraded replies after maintenance and list requests hit HTTP read timeouts at 5,020.529 and 5,015.645 ms. The corresponding maintenance dispatch reached its ten-second timeout. The failure occurred earlier than the original installed run, so this establishes a timeout mechanism rather than proving the cause of every earlier failed read.

The three slowest single-fact retrieval metadata updates took 4,238.240, 3,822.417, and 3,115.817 ms, accounting for nearly all of their engine retrieval duration. HTTP response writes were brief. These measurements locate a storage-update delay but do not yet separate SQL execution, locking, and commit time. No timeout, durability setting, or acceptance gate was relaxed.

## Invalid diagnostic setup (v1)

The first diagnostic put its fixture beneath this Git checkout. Both test projects therefore resolved to the same Git identity, invalidating the isolation assertions. Read-only database inspection confirms that the corrected fact and canary shared a project ID. Its final origin check also incorrectly rejected a namespace package without a file origin. This run is excluded from product correctness and timing conclusions. Raw output and the executed helper remain unchanged.

The original failed installed run remains in [combined-package-checkpoint-v2](../combined-package-checkpoint-v2/README.md). Its pytest temporary databases were no longer present when inspected; no claim about their surviving fact contents is made. The dedicated diagnostic databases remain outside the archived reports, with read-only observations and file hashes recorded here.
