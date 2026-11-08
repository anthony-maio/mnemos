# Identity cache adversarial check

Worker integration remains withheld. A real-Git regression showed that a repository using `GIT_OBJECT_DIRECTORY` could become invalid when its external object directory disappeared, while the prototype returned a cached identity. The environment string and repository routing files had not changed. Authoritative resolution rejected the same repository.

The prototype now bypasses caching for unknown Git environment selectors. Configuration selectors remain covered by the caller's required external-policy generation, and selectors explicitly removed by the authoritative resolver retain their existing behavior. This fixes the reproduced external-object-directory case without trying to infer the meaning of future Git selectors.

The first adversarial run reproduced the stale result. The next run fixed it but exposed two test assumptions: an inherited `GIT_PAGER` conservatively disabled caching, so tests expecting a hit did not exercise the intended condition. The synthetic repository fixture now isolates inherited Git variables. A separate test verifies that an unknown selector delegates on every lookup. The final run passed 37 tests across cache behavior, project identity resolution and scope contexts. Strict type checking passed.

This is not evidence that a policy-generation provider exists or that the cache is ready for host use. The provider remains unimplemented, and the prototype remains disconnected from adapters, installed packages and live pilots. No latency gain is claimed. Existing timing failures remain unchanged.

Next: investigate the measured interval between dispatch completion and service-submit return, preserving fresh Git resolution, registry validation, fsync and permission enforcement. Return to cache integration only with a complete external-policy invalidation contract and corresponding tests.
