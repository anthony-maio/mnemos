# Codex v9: inspection identifier mismatch

The fresh native attempt failed during its first restart recall. Codex called
`mnemos_inspect` with `fact_id`; the MCP tool required `chunk_id` and returned a
validation error. The host corrected the argument and inspection succeeded,
but the strict validator rejects a phase containing any unsuccessful tool
response. That rejection and the original trace remain unchanged.

The attempt used product `cddf8adc`, wheel SHA-256
`c6f6ff3ddf260a2a736867f01ce923879d68e2dd91ff44abbf0f7a22e1e6982f`,
and harness `a2fb3043`. The [v9 protocol](../../plans/2026-11-07-native-codex-inventory-v9.md)
and six original lifecycle prompt hashes were committed before inference.
Codex CLI 0.164.0-alpha.11.1, its Code Mode binary, `gpt-6-astra`, high
reasoning effort and the 180-second phase limit matched v8. There was one
attempt, with fresh native homes and storage, and no corrective message.

The inventory amendment has 34 new checks, covering complete observations,
bounded startup readiness, fabricated or filtered results, mismatched IDs,
overlapping cells, changed payloads and changed lifecycle prompts. All 179
focused checks passed; Black and strict type checking passed. These checks
validate the harness, not the native lifecycle outcome.

| Phase | Result | Evidence |
| --- | --- | --- |
| Teach | Passed | Exact advertised inventory after two observations; three correlated calls; original job completed |
| First restart recall | Failed | Retrieve succeeded; inspection with `fact_id` failed; inspection with `chunk_id` succeeded |
| Correct | Not run | Attempt stopped after the failed phase |
| Second restart recall | Not run | No correction was performed |
| Forget | Not run | No deletion was performed |
| Final restart recall | Not run | No source replay was attempted |

An independent audit revalidated the teaching phase and confirmed the failed
inspection from the original native response. It made no model calls. The
original fact remained at version one, and SQLite integrity and foreign-key
checks passed. The failed phase does not receive a complete inventory receipt
because the strict trace parser rejects its unsuccessful tool response first.

Cleanup passed. All six recorded process identities were absent, runtime
descriptors and copied credentials were removed, and normal authentication,
configuration and embedding assets retained their hashes. The private archive
contains 340 verified files, including the native streams, database, preparation
and failure audit. SHA-256:
`9b5320cc5dc4ccd2604e1812dbc050d7976609b7f19112fcec43d36d6d01aeb6`.

The service and Hermes already accept `fact_id`. This failure motivates a
compatible MCP alias with explicit conflict rejection. A follow-up fix needs
its own installed and native validation; it cannot change this result. The
pilot remains inactive with zero days. No live configuration or publication
changed.

- [Sanitized receipt](native-codex-v9-receipt.json)
- [Previous lifecycle and catalog limitation](native-codex-v8.md)
