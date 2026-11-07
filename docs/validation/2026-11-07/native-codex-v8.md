# Current Codex lifecycle: six phases pass, catalog check unverified

The current installed artifact passes the six-phase explicit-memory workflow
with 19 correlated native tool calls. The complete v8 protocol remains
incomplete: its requirement to verify the exact advertised tool set is not
established by the retained logs. Configured permissions and observed calls
are verified, but some phases only retained discovery queries filtered for
the requested operations. Later queries in three phases covered the Mnemos
namespace; the full inventory remains unverified across all six phases.

The attempt ran once on product `cddf8adc`, wheel SHA-256
`c6f6ff3ddf260a2a736867f01ce923879d68e2dd91ff44abbf0f7a22e1e6982f`.
The [protocol](../../plans/2026-11-07-native-codex-current-v8.md) was committed
at `c70b6c7a` before inference. Codex CLI 0.164.0-alpha.11.1, its Code Mode
executable and `gpt-6-astra` matched the pinned runtime. Each phase used high
reasoning effort, one prompt, a fresh native home and the original 180-second
deadline. There were no retries or corrective messages.

The [installed CLI/MCP check](installed-smoke-cddf8adc.json) passed first,
including real BGE embeddings, evidence inspection, recall after a service
restart and foreign-project isolation. A fresh external native installation
then matched all 95 package files and the expected 57 dependency versions.
Seventeen embedding assets retained their hashes. Product code, tests, examples
and native harnesses were unchanged from the full regression checkpoint with
3,495 passes and four platform skips.

| Phase | Native calls | Verified behavior |
| --- | ---: | --- |
| Teach | 3 | One explicit store; the original job reached completed |
| First restart recall | 2 | Same asserted fact, version one and retained evidence |
| Correct | 6 | Inspected the same ID/version; the correction job completed |
| Second restart recall | 2 | Same fact ID, version two and corrected content |
| Forget | 4 | Reviewed preview and exact selection; deleted dependent records |
| Final restart recall | 2 | Forgotten content absent; retained review convention present |

An independent audit rebuilt all 19 normalized calls from native model, code
cell and MCP runtime IDs, checked the observed model and prompt for all six
sessions, and reapplied the lifecycle assertions. It verified one exact original
source replay was suppressed after deletion. Three retained, unrelated and
foreign controls stayed unchanged. Forgotten labels were absent from the final
database and fresh native context. SQLite integrity and foreign-key checks
passed without changing database bytes. The audit made no model calls.

A separate catalog audit records the missing observation. The original runner
result remains `passed` for its six implemented phase checks; the sanitized
receipt reports `incomplete` for the full prospective protocol. The next
validation must capture the complete Mnemos inventory before memory calls in
each phase and compare it with that phase's configured allowlist. This result
must not be silently upgraded by rerunning the existing checker.

All 13 recorded process identities were absent after cleanup. Runtime descriptors
and copied credentials were removed; normal authentication and configuration
bytes were unchanged. The verified private archive retains 804 files, including
the raw native streams, prompts, database, both audits, preparation logs and
wheel. SHA-256:
`8fefc453d89e4ec28a287d97e6ac5788c12754b76e9e075e333002cca1935b3d`.

This is one synthetic lifecycle on the named artifact. It does not validate
the opt-in resident hook launcher or drafted pilot configuration, establish
answer-support or extraction quality, meet the failed 100 ms hook target, or
count as daily use. The private pilot remains inactive with zero days. No live
configuration or publication changed.

- [Sanitized receipt](native-codex-v8-receipt.json)
- [Full regression checkpoint](full-regression-native-hook-v1.md)
- [Complete hook performance comparison](native-hook-performance-v1.md)
