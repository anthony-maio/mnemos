# 0.6.1rc1 review checklist

This is a local prerelease candidate, not an approved or published release.

- [x] Preserve graph rollback and caller-owned transactions.
- [x] Integrate content-recency and Windows plugin interpreter repairs.
- [x] Verify combined source at b3508f6d: 353 tests; strict typing and formatting.
- [x] Assign 0.6.1rc1 consistently to package, runtime and plugin manifests.
- [x] Document schema-3 incompatibility and rollback limits.
- [ ] Build and verify the exact prerelease source archive and wheel.
- [ ] Verify installed migration, transaction behavior and MCP roundtrips.
- [ ] Review the supported Python/platform matrix for this exact artifact.
- [ ] Resolve or explicitly review the original legacy benchmark failure; a
      different development comparison cannot overwrite that result.
- [ ] Complete interactive native-host and actual pilot obligations before
      claiming them. Automated stdio configuration tests are separate evidence.
- [ ] Approve the exact upstream diff, artifact hashes and upgrade plan.
- [ ] Merge, tag, publish and verify the published artifact only after approval.

[Upgrade instructions](legacy-upgrade.md) require older clients stopped before
schema migration. Do not restore a pre-upgrade backup after new writes or forget
requests without reconciling those changes. There is no automatic downgrade.

The full six-priority improvement goal remains open. This checklist does not
waive semantic quality, complete-command latency or two-week pilot requirements.
