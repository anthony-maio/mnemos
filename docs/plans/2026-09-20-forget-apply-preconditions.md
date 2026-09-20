# Return copyable reviewed-forget preconditions

## Evidence and scope

Native Hermes v13 on product `db21750` completed zero of three full lifecycles.
Two attempts supplied `preview_limit=20` after previews returned `limit=100`,
despite copying the digest correctly. The database correctly rejected these
requests. The third attempt selected source IDs outside the selectable list.
These failures remain in their original archive and are not repaired by a
benchmark wrapper.

The source inspection also found a separate MCP advertising defect: the
preview function's first docstring described single-fact deletion, hiding the
second string with the intended preview description. A test against the
registered FastMCP tool reproduces it. Correct the description to state that
preview is read-only and explain how its preconditions are used.

The preview currently returns the fact ID and version under `target`, the
limit as `limit`, and the digest as `preview_digest`. The apply tool instead
expects four top-level fields, including `expected_version` and
`preview_limit`. Returning a ready-to-copy set of preconditions removes this
translation step. This is an interface improvement hypothesis, not evidence
that native model reliability has improved.

## Chosen change

Add `apply_preconditions` to a complete storage preview, containing exactly
`fact_id`, `expected_version`, `preview_digest`, and `preview_limit`, with
values from that same preview snapshot. Return null for an incomplete preview.
Keep all existing response fields and request schemas. Both Hermes and MCP
guidance tell callers to copy these fields unchanged, then supply explicitly
reviewed fact/source selections from the preview.

Do not generate a deletion selection, add defaults to the selection, repair
incorrect arguments, cache a preview in an adapter, alter digest construction,
or weaken transactional scope/version/dependency checks. A complete preview
does not imply every possible selection is valid, or that all semantic copies
were discovered. Existing clients remain compatible.

A persistent review-token design would add receipt storage, expiry, replay,
and migration responsibilities. A stateless token would require a new parser
and request contract. Neither is needed to remove field-name translation.
Guidance alone leaves that translation in place, so it is not the selected
approach. No live configuration or database migration is involved.

## Verification

First add a failing test that copies the returned preconditions into the
existing Hermes apply tool, adds reviewed selections, and verifies deletion
and suppression at both default and non-default limits. Test incomplete
previews return no applicable preconditions. Exercise stale captures and a
changed limit through the same copied object; both must reject atomically.
Keep existing foreign-scope, selection, and dependency tests unchanged.

Run focused reviewed-forget, service, MCP, adapter, and documentation checks,
then package/installed verification for the changed product. Only after a
verified product checkpoint should a prospectively frozen native successor
test whether the interface changes observed behavior. Preserve the original
prompts, capture, phase deadlines, and all stored-state erasure requirements.
No native reliability claim follows from the deterministic tests alone.
