# SQLite worker queue integration

Status: isolated source and installed lifecycle verified. No command timing acceptance, promotion, publication or live activation is claimed.

The experimental Codex worker accepts `--queue-captures --outbox-backend sqlite`. JSON remains the default. The runtime descriptor identifies the selected backend and refuses attachment to an incompatible existing owner; descriptors predating this field retain JSON compatibility. SQLite selection owns its client and connection rather than changing an injected client's queue implicitly.

The selected queue has its normal replay pump. A separate compatibility pump drains JSON fallback backlog when SQLite is selected, including records added after startup. Switching back to a queued JSON worker also drains an existing SQLite queue. The implementation leaves each format in its own file and relies on the service's original-event idempotence when both contain the same capture. It performs no automatic migration or bulk deletion. Shutdown stops both pumps, closes their transports and then closes the owned SQLite connection.

Forty-five source integration checks passed. Both JSON and SQLite ran the real-service scenario: queued capture, restart delivery, racing duplicate sends, completed processing, same-ID correction, forgetting, separate-process suppression inspection, and replay after another restart. The SQLite case reintroduces both JSON and SQLite copies after forgetting and confirms no fact, pending job or episodic event is restored. Additional checks cover incompatible-owner rejection, invalid mode selection, connection/pump shutdown, timer-driven JSON backlog arriving after startup and rollback to JSON with duplicate backlog across both formats. Strict type checks passed for the worker and SQLite outbox modules.

A new wheel and sdist were built from `4928d90a0eee6a7798d029a86927755a8fa40fbf`, then installed into a fresh Python 3.12.11 environment with the prior dependency versions. All 127 installed checks passed. One test skipped because Windows prevents renaming an open SQLite database; replacement-path behavior on a platform permitting that operation remains unverified. The runner verifies imports come from the isolated installation. All sixteen copied test files matched source hashes.

The wheel's 99 package source/resource files matched committed source and installed copies, including after tests. The sdist's 106 source members matched source, with required example and smoke resources present. The hash-verified evidence archive includes both artifacts, dependency constraints, copied tests, JUnit, parity checks and helpers. See the [receipt](sqlite-worker-integration-v1.json).

Wheel SHA256: `dbb38d06c3b2151deac8b72a5c5d8df73c9fa05525bf5056ec31174547b20590`.

The earlier component tradeoff remains: SQLite append was faster, but acknowledgment and most complete append/delete cycles were slower. This installed validation does not establish a faster complete hook command, cold-start latency, scale, native-host behavior or power-loss guarantees. A prospective command comparison must use the same installed wheel, identical launcher and isolated queue/service state for each backend, preserve every sample and include compatibility-pump overhead. The original 100 ms requirement remains unchanged.

The previous installed package, native binaries, frozen measurements and optional pilot are unchanged. Canonical delivery still does not select this candidate. Quality, fresh acceptance and actual daily-use obligations remain open.
