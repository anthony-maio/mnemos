# Spreading visit-budget source/package checkpoint

Frozen source 995d9c2 passes 2,158 tests with
4 skips and all six source/package checks. The four standalone
visit-budget cases, nine support-revision cases and ten experimental inventory
v3 cases are included and pass. Formatting, Python 3.10 type checking,
JavaScript syntax, build and package metadata checks pass.

Standalone spreading calls now honor the configured max_visited when no call
override is supplied. The managed engine already passed that limit explicitly;
this does not supersede the real-model graph recall negatives. See the sibling
[repair record](../spreading-visit-budget-v1.md) for the reproduced baseline
failures, controls and focused checks.

The wheel changes production code and needs its own installed verification.
The separate extraction experiment remains unadopted and uses its own frozen
candidate/provider source. Semantic quality, full-module benefit, wider host
recovery and daily-use acceptance remain open. No live configuration, migration
or publication changed.
