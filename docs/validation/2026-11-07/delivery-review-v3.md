# Frozen candidate delivery bundle

The updated bundle reconstructs delivery target `fb062fe3ff11dbbbae33a08a2736f40b15b05585` from
base `9770c009`. A private Git index verifies the exact target tree. The patch
contains 478 tracked changes, including 35 product/README files and the completed
support, storage-package and adversarial-review reports. The real delivery
checkout and website remain unchanged; this is prepared integration.

The product remains `f5a944ea`. Its original verified wheel is included unchanged.
A new source archive builds a wheel whose 101 members all match that original
wheel byte-for-byte, including package resources and distribution metadata.
Twine checks pass. The source archive uses the product commit's documentation;
the delivery tree includes newer verification documentation at `fb062fe3`.
These two documentation bindings are explicit in the receipt.

The first Git-export build differed in line endings in 11 Python files and the
resulting wheel RECORD hashes. The independent comparison confirmed identical
content after CRLF normalization. A second private build restores only those
files' existing wheel line endings, then verifies exact wheel-member equality.
The first attempt, diagnostic and both build helpers are retained. No product
file or existing installed wheel was changed to obtain this result.

The 3,674-pass full suite remains bound to `1299ed9b`, with four platform skips.
Package source is unchanged. The later named codec has 132 focused checks,
and the current documentation has 12 checks. This bundle does not claim a new
full-suite run. Native Codex acceptance remains on `c80eb051`, and neither that
pass nor experimental support results qualify the current pilot automatically.

- Patch SHA-256: `047e678c38eb367e3acacc2948f06bd28648d35cb4d6f160ae36c5117276271b`.
- Original wheel SHA-256: `07c5b424bc7731cacbebb45f44709b3f19509c2cde3150a6451c92d12b27e6ae`.
- Source archive SHA-256: `97a201103f96e1f12b6034c58ad7a28f22bc2da874e137b040251c3a6cd27f3c`.
- Private archive SHA-256: `e40189fc92fe80c92baa6f05ef2d798d4497f81281eb917d3ee4631a896c59d7` (14 verified members).

The [receipt](delivery-review-v3-receipt.json) supersedes the older
[delivery preparation](delivery-review-v2.md) for the frozen product. This report
and its receipt are outside the bundle's target tree; their later addition does
not change its package identity or require another package rebuild. Refresh the
bundle only for a deliberate source or delivery-content change.

Public master was rechecked at `86644437`; it was not modified. Native acceptance
on the selected configuration, quality, durable acknowledgment timing, actual
pilot use, canonical integration and website/public delivery remain open. The
pilot is inactive at zero days. No publication or live configuration changed.
