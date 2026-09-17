# Provisional native Claude v8

This folder preserves the bounded v8 native Claude run against source commit `a38ee70d19f2ce8413d1ab7c9ce2ff26ffd07b46`. The installed host was Claude Code `2.1.191`, using model `claude-sonnet-4-6`. The verified wheel SHA256 was `94b19bb5fd758e715031a8a6d6e3599c0a6955145e5717bf9325607900096ecc`.

The native process returned zero in 9,727.9 ms. It was marked `not_observed` because the original raw duplicate gate failed: 3 captured events were present, with 2 user events and 1 assistant event. The original gate remains false even though the separate provisional identity contract passed.

The separate identity diagnostic found 2 normal events and 1 `provisional_unassociated` event. It found one normal transcript-associated user and one normal transcript-associated assistant. The missing-ID fallback stayed `deferred` with a `capture_only` job, and after one fixture maintenance the job was completed without fact metadata or evidence derived from the provisional event.

The durable checks passed for restart survival and Stop replay without extra event IDs. The run used constant embedding vectors, disabled cognition, and a synthetic codename prompt. It provides host capture, identity routing, and durability evidence. It provides no model quality or semantic retrieval proof. The observed cost was `$0.025425`.

The run supplied disposable hook settings through Claude's `--settings` argument and consumed a generated project MCP configuration. It therefore exercises those explicit files and does not prove Claude plugin auto-discovery or loader behavior.

The raw hook traces, native stream, transcript, and checkpoint are synthetic disposable-fixture diagnostics. Settings, MCP configuration, scope registry, environment files, database, and credentials are excluded. The complete copied set is indexed by relative path, byte count, and SHA256 in `evidence-index.json`.
