# Updated delivery review preparation

The private review bundle now reconstructs `0456cb5d8e6622050a5c139c2722e691b6b93f2a`
from delivery base `9770c009f8d3b4919ecfd18810d700badc53c638`. Applying its patch to a
private Git index produced the exact target tree. The real indexes, branch
heads, delivery checkout and website stayed unchanged.

The patch covers 427 tracked files, including
35 product or README files. It includes
the current hook transport, retrieval changes, host guidance, inspection
alias and Windows descriptor fix. The wheel matches product
`3d1866d50d6df4d46666155006792ad46a9c6f11` and its installed verification receipt.

Preparation occurred while the full suite was running. The subsequent
[full regression](full-regression-descriptor-sharing-v1.md) passed 3,560
cases with four platform skips on this same target. The bundle preserves
its preparation-time manifest; its patch does not contain the later
full-suite report or this preparation report. A new source archive was not
built for this bundle. These limits remain explicit in the
[receipt](delivery-review-v2-receipt.json).

This supersedes the older [749f1213 review preparation](delivery-review-v1.md).
It is a patch and installed-wheel review bundle, with no applied integration,
public-branch refresh or transferred native-host or quality acceptance.
The native v11 pass remains on c80eb051. Final integration must include the
later reports and bind resulting artifacts to their actual source.

- Target tree: `7c19c821b5644944e8a4f7088f25389db686952e`.
- Patch SHA-256: `d126cf5f502ba67f3cb1db272ef1ba3d01096e5df07f93406746cdcb47bf4c68`.
- Wheel SHA-256: `253ca806dba5aa817bd7ddcba1af123535b29853aa211fee61a14f71799a62b8`.
- Private archive: 5 verified members,
  2,759,439 bytes.
- Archive SHA-256: `6cd6b660d7f7e900a1af1e3404451704d9b1b2c27f74ce487839fbdd26fcce18`.

The canonical website and delivery checkout remain at their preserved
heads. The pilot stays inactive at zero days. Nothing was published.
