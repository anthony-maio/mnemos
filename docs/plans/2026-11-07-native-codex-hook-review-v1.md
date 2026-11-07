# Native Codex hook review on the frozen wheel

Use the existing product `f5a944ea` and wheel SHA-256
`07c5b424bc7731cacbebb45f44709b3f19509c2cde3150a6451c92d12b27e6ae`.
This step establishes normal persisted hook trust in a disposable profile.
It does not activate the private pilot or establish hook execution, retrieval
quality, capture timing or daily use.

Prepare a separate Codex home, empty project, scope registry and memory paths.
Take the six definitions from the current private pilot patch, changing only
the disposable launcher path. Bind the installed package, native executables,
helpers, settings and hook definitions by hash. Keep normal host files and the
inactive pilot unchanged. Use the installed Codex binary already bound to the
current native lifecycle check.

Read `hooks/list` through a short-lived app server using only initialization
and inventory methods. Require exactly the six expected user hooks with their
original commands, matchers, timeouts and context limits. Reject extra hooks,
different sources, changed hashes and disabled or managed definitions.
Configured, modified and untrusted hooks do not establish normal trust.

The user reviews those definitions in the native `/hooks` UI. The launcher
uses normal trust settings, a read-only sandbox and the isolated profile. The
MCP server is disabled for this review. It copies the existing login only after
preflight, removes that copy on exit, and stops any disposable service started
by SessionEnd. Preserve original credentials and configuration. Submit no
model prompt; any recorded turn requires inspection rather than a review pass.

After exit, read a fresh inventory and require the same hashes with all six
reported as trusted. Verify process cleanup and unchanged original files.
Partial review stays pending. Do not edit a trust store or use a trust bypass.
Preserve failed preparation and review attempts with their original status.

The subsequent execution check must use these same trusted definitions and
the frozen installed wheel. Freeze that execution protocol before starting a
native model turn. A trust-only pass cannot substitute for that acceptance.

Sources: [Codex hook review](https://learn.chatgpt.com/docs/hooks#review-and-trust-hooks),
[current pilot preparation](../validation/2026-11-07/persistent-pilot-v4.md), and
[candidate convergence](2026-11-07-candidate-convergence.md).
