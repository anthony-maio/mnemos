# Native Claude scoped identity check

Claude Code 2.1.191 ran against the installed `20f271d` wheel in a private,
disposable project. Invocation-specific settings loaded SessionStart,
UserPromptSubmit, and Stop wrappers. The command selected Sonnet 4-6,
project-only settings, no tools or MCP servers, no session persistence, and
an $0.08 budget. No live host configuration changed.

The native turn completed in 6.644 seconds for $0.009372 and returned the
random canary stored only in the fixture memory:
`The private release codename for this project is **mnemos-7f73ced73ac9**.`
The prompt did not contain that value. The parsed stream had no tool use.
Both conversation roles were captured with `claude:v2:` source identities,
and both events survived service restart. Source identities are preserved in
event metadata; SQLite's internal event IDs are separate hashes.

All three hook wrappers returned zero. SessionStart and UserPromptSubmit
returned memory context. UserPromptSubmit and Stop also emitted degraded
adapter diagnostics. The native payloads lacked a stable prompt ID and
provided transcript paths; this no-persistence run did not establish usable
transcript coverage. The runner deliberately did not inspect external host
transcript files. Fallback capture is observed, but identical repeated messages
without native IDs or transcripts remain ambiguous. This does not close full
transcript or native correction/forget acceptance.

The fixture used constant unit vectors, unavailable cognition, and disabled
biomimetic modules. This tests native hook delivery, scoped source identities,
and restart persistence. It does not measure semantic retrieval or extraction
quality. The installed-wheel two-project identity/forget checks are separate
evidence in `../claude-scope-verification/attempt-v1/installed-wheel.json`.

The wheel SHA256 is
`0ecaa634fe447621d6366b93accad7abfa2495063d9b425e6043973ea7b86663`.
All 53 tracked package files match the pinned source after line-ending
normalization, and all 56 packaged Mnemos members match the wheel bytes.
This is also the byte-identical wheel from the stable full verification.

The original report and exact runner are preserved. Reviewed synthetic hook
records, answer and usage observations, durable events, and cleanup evidence
are indexed alongside them. Host init and account rate-limit records are not
exported. The runtime descriptor is absent, no fixture process remained at the
cleanup check, and no authentication copy was made. The private fixture remains
available for diagnosis.
