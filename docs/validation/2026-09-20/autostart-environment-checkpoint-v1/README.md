# Autostart environment package checkpoint

Frozen product `a64afb73d1b99510cfa32e96164bf6c11458b51e` passes 1,834 source tests
with 4 skips, all six source/package checks, 703
isolated installed cases with 4 matching skips, and seven replay
assertions. The installed checks compare individual case outcomes, loaded
module origins, and package bytes against the frozen source and wheel.

Wheel: 305,087 bytes; SHA-256 `594c7cf6888ae919e9bd99d35a7484aa3b6e2e0609c50ae5b5d806a138b30273`.

The [development record](../autostart-environment-development-v1/README.md)
records the reproduced client and test-isolation defects and 139 focused
passes with two skips. Default clients now honor `MNEMOS_AUTO_START`, including
the bridge and managed CLI. Explicit library arguments retain precedence;
stopping a service never autostarts one. Disabled clients still queue capture
and can replay to an existing owner. Disposable host phases independently
pin their database and runtime locations.

The [native v1 failure](../native-claude-outage-failure-v1/README.md) is retained.
An unexpected service wrote three synthetic events and three jobs to the
default database, with zero scoped facts observed. Its owner is stopped,
the private backup is preserved, and no cleanup deletion has been performed.
These tests do not turn that failed attempt into outage recovery evidence.

The new outage acceptance helpers and runner postdate this product checkpoint.
They have separate focused checks and a [frozen successor protocol](../../../plans/2026-09-20-native-claude-service-outage-v2.md).
Their additions do not change the product package. Native outage recovery,
full-module quality, Desktop, and the operational pilot remain separate gates.

The evidence index binds the raw files; this README is outside that index.
The earlier failure/development archives had checkout line-ending conversion
corrected in `c9ae826`; their twenty indexed files match the committed bytes.
