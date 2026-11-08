# Storage repair merged after hosted validation

With explicit user approval, [PR #7](https://github.com/anthony-maio/mnemos/pull/7) merged at `47eba2e1263bddf91204b59a59ca605ff34e42ec`, followed by [PR #8](https://github.com/anthony-maio/mnemos/pull/8) at `588a190b062dab98f714c37b55d2292d1725b409`. Remote master was fetched and verified at the second commit. Its tree `cceba00ac8ec9af1d1de3b24464289aad7e9714c` exactly matches the final tested candidate. No package, tag or release was published, and live host installations remain unchanged.

The existing CI only selected PRs targeting master. Commit `79435de9` removes that PR branch filter to cover stacked PRs; master push behavior and every job definition, command and benchmark threshold remain unchanged. GitHub's [workflow syntax documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax) describes this target-branch filter. The first job-section comparison encountered Windows line-ending differences; comparing normalized lines confirmed identical jobs.

## Hosted results and merge sequence

The [stacked run](https://github.com/anthony-maio/mnemos/actions/runs/42501333385) passed all eight jobs on head `79435de91b1c6f2c22967d0d6bc69f9cde00e11a` against base `9c9ebf84`. PR #7's own run `42370536244` was also confirmed successful before merging its exact head.

After #7 merged, #8 was retargeted to master and marked ready for review. Merging the new master into its branch produced integration head `15499b03fe8a0d5ec251009b197e9891c37c64f3`; Git tree comparison proved that this changed no candidate files. The [final integration run](https://github.com/anthony-maio/mnemos/actions/runs/42501658514) then passed every job against master `47eba2e1` before the merge was requested with the exact expected head SHA.

| Final integration job | Result |
| --- | --- |
| Python 3.10, 3.11 and 3.12, Ubuntu | 325 tests passed on each version |
| Lint | Black: 66 files; mypy: 31 source files; both pass |
| Legacy benchmark gate | Both repetitions pass with unchanged datasets and thresholds |
| Tier 1 MCP | 4 automated checks pass |
| Tier 2 MCP | 1 automated check passes |
| Release-readiness job | 12 documentation/MCP checks pass; checklist file present |

The log checkout records bind the final run to GitHub's merge preview of `15499b03` into `47eba2e1`. These automated MCP tests do not establish interactive native-host acceptance. The job named release_readiness checks documentation/MCP behavior and file presence; it does not prove all checklist items, actual two-week use or package publication. The legacy benchmark is distinct from the original supported-answer and complete-latency gates, which remain unresolved.

## Preserved evidence

- Initial eight job logs and job metadata: `.venv/storage-pr8-ci-evidence-v1.zip`, SHA256 `e7aea4bb8d1d156e10e4f38763236cdd14a429c6c55930e2c26be61c6313e09d`.
- Final eight job logs and job metadata: `.venv/storage-pr8-integration-ci-evidence-v1.zip`, SHA256 `3df4c9f72ae4f641449a411dc967ace274bebc281f935a76505f7dc6ad25ef8f`.

Each archive has ten members, including its hash manifest, and every manifest member passed readback. Earlier failed candidates and their reports remain unchanged. This completes upstream acceptance of the transaction repair and its compatibility fixes. Affected-user package delivery and the remaining five improvement priorities are still outstanding.
