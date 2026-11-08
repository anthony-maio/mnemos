# Local canonical delivery integration

The local canonical code and website now contain the frozen product and its
current evidence. Code `ebec906a` and website `1565bf45` pass the checks below.
Both checkouts were clean before and after their fast-forward. Nothing was
pushed, released, deployed or enabled in a live host.

The product remains `f5a944ea`. This integration updates delivery documentation
and includes later tests and reports. Three development-helper changes resolve
strict typing: an explicit benchmark package root, a streaming SHA-256 helper
compatible with the declared Python target, and guards around optional native
dispatcher and descriptor state. They do not change the frozen product, model
prompts, accepted facts or evaluation scores.

| Check | Recorded result |
| --- | --- |
| Full isolated Python suite | 3,758 passed, four platform skips, 3,762 collected |
| Native-launcher cases within that suite | 75 passed |
| Strict type checking | 228 source files passed |
| Python formatting | 545 files passed; notebook formatting was not run |
| Environment binding | 86 pinned dependency versions, plus package and bootstrap pip |
| Historical fixtures | All 208 hashes unchanged |
| Local website | Desktop and mobile navigation passed, no page errors or body overflow |
| Receipt downloads | Three receipts in each viewport, all six byte matches |
| Preserved wheel comparison | 96 package files match after eleven newline-only normalizations |
| Watched live configuration and pilot paths | All fourteen unchanged |

The final run uses a dedicated environment and verifies that both product and
benchmark imports resolve to the integration checkout. The first full run
passed its tests but failed typing and has an import-path qualification limit;
it is retained as diagnostic evidence. The baseline documentation failure and
the initial environment inventory assertion are also preserved. Only the final
isolated run supports full-suite acceptance. This was one local CPython 3.12 run,
not verification across the full supported Python or operating-system matrix.

The homepage and evaluation page now state the unpublished candidate's current
limits. Earlier results remain dated and bound to their original artifacts.
The two checkpoint screenshots were inspected for readability. Browser checks
used a temporary local server, which stopped afterward; they do not verify the
hosted website. The public installation command still selects the older public
preview branch. It does not silently install this local candidate.

Exact code and website review patches reconstruct their target trees from
`9770c009` and `08d393ee`, respectively, in private indexes. Real indexes and
working trees stayed unchanged during reconstruction. The subsequent local
fast-forwards followed fresh clean-tree checks and a scan of known host/pilot
configuration and current matching process command lines. That scan found no
canonical-checkout consumers, but does not enumerate every possible consumer.

The existing wheel was not rebuilt. Its packaged documentation remains bound
to `f5a944ea`; the repository and website carry the later reports and limits.
Package version `0.7.0b1` and plugin version `0.7.0-beta.1` retain their existing
format-specific alignment. The public PyPI inventory remains separately dated.

The [receipt](canonical-integration-v1-receipt.json) binds the full revisions,
patches, test output and private archive. The archive has 51
members and SHA-256 `a92c02dc0bbfe6fc9b9773224fea70e66fba7d3afc27abdbc4ae1dea4334ce35`. It preserves the earlier
failed attempts, completed checks, local integration record, browser captures,
patch reconstruction and offline PR draft. Virtual environments and private
Git indexes are excluded. This report and its documentation links are a later
documentation-only follow-up to the tested code revision.

Quality and timing requirements still fail. The [closed support comparison](retrieval-support-development-v1.md)
does not qualify default retrieval, and the [installed timing result](installed-pilot-command-timing-v1.md)
does not pass complete recall or durable host acknowledgment limits. The prior
native Codex lifecycle retains its hooks-disabled binding; no native-host
acceptance was repeated here. The two-project, fourteen-day pilot remains at
zero days. The optional learning-pilot proposal remains unapplied. These local
integration checks do not make either package candidate release-ready.
