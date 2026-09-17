# Atomic schema repair verification

Frozen source `9e253390c58302817a18b6504f55899b36a2c27b` passed the complete
verification run on September 17, 2026. Pytest reported 703 passes and two
optional WSL skips. Black, strict mypy with the compatible Python 3.10 checker,
JavaScript syntax, wheel/sdist build, and Twine checks passed. The run began
at 09:29:43 UTC and completed at 09:54:21 UTC. The earlier source236 startup
failure remains in the [previous verification record](../hermes-registry-verification/README.md).

The wheel is 264,473 bytes, with SHA256
`65f0354fd97b6a492a7c5dde2bf0f8eb008c3ff21b81842631e0f9a92c545a62`.
The source distribution is 2,435,750 bytes, with SHA256
`2f3d64e256fd71e6ed112b33b0b5dfb85a2e24c9d53c323f5fb94200e62400c5`.
Exact commands, logs, artifact locations, and runner bytes are preserved here.

A fresh Windows environment installed that exact wheel and passed all seven
Claude replay assertions. The probe checked installed package bytes against
the wheel and normalized source. This is an installed-package SQLite replay
check, not a new native Claude conversation.

The actual Hermes 0.21.3 interpreter in WSL discovered the optional provider
through the extracted wheel's entry point and reported it available. The
observed Hermes revision was `36842e639a178741b70c821e0f4cdd129cd0dfe3`.
All 53 source files and three packaged assets were verified. Network and
service call counters were empty. Extraction used a disposable directory and
did not install into or change the live Hermes environment. This proves wheel
entry-point discovery, not a completed native model turn. The saved payload
has a `.txt` suffix to keep diagnostic code out of repository checks.

The Codex lifecycle runner passed its installed-wheel API preflight with an
isolated interpreter, constant fixture vectors, unavailable cognition, and
explicit remember/maintenance/retrieve assertions. It ran CLI capability
checks but no native model turn. The first preflight stopped before those
checks because the configured v8 reference digest contained two extra hex
characters. The corrected value matches both the original v8 report and its
unchanged helper. The failed runner bytes were reconstructed by reversing
that one literal change and verified against the failed report's runner hash.
Both preflight reports and exact matching runners are retained.

Full native lifecycle acceptance remains open. The separate
[raw-capture overlap case](../capture-forget-overlap/v2/README.md) shows why
passing this suite does not establish that deleting one fact removes every
captured copy of its assertion. Previewed related-record removal is being
implemented against that preserved case. No release or live migration was
performed by these checks.
