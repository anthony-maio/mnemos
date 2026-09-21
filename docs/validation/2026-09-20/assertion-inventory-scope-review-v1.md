# Source inventory scope repair

Independent review found that the first candidate accepted a missing profile
identity or contradictory scope metadata. This was a reusable validator defect;
all 46 frozen development inputs have consistent identities and also pass the
stronger check. The running v1 experiment and its archived preparation remain
byte-identical. The repaired validator is not substituted into that run.

The repair requires a nonblank profile, valid canonical scope and scope ID,
and consistency with optional scope metadata. For project/workspace scopes,
the corresponding optional context ID must match the canonical scope ID.
Other project/workspace context can remain present for workspace/global events;
this check does not prove registry membership or authorization to write.

Ten new invalid-input tests failed before the repair. Three valid-context
controls passed before it. All 29 candidate tests now pass; Black and Python
3.10 mypy pass. A separate offline pass confirms all 46 frozen inputs pass the
repaired validator, and every preparation-indexed file still matches its hash.
No generation calls were made for this verification.

The first preparation binds itself to an independently archived, blob-verified
artifact. Its standalone runner does not carry an external expected preparation
hash. That workflow limitation remains explicit; the archive and terminal
collector provide the external binding. Quote matching remains a provenance
structure check, not a semantic-entailment or completeness guarantee.
