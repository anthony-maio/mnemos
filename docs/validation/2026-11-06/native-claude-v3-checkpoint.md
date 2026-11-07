# Claude v3: forget preview exceeds inline output

The new attempt failed. Teaching, restart recall, same-ID correction and
version-two recall passed. Forgetting failed, and final restart/replay was not
attempted. The original failed result remains unchanged in the private archive.

Product `5de8b1f1` built wheel/source artifacts, passed Twine and passed a fresh
[installed smoke check](installed-smoke-5de8b1f1.json). Its wheel SHA-256 is
`7781fcc3b38f68ad7a6171074701dd1e539eecc9820178343efec93b047819e1`.
The installed catalog contains the clarified forget-selection descriptions;
all 57 installed package versions match the preceding native environment.
Runner `67166200` changed only the prospective amendment binding. Native phase
prompts, budgets, parser and lifecycle assertions were unchanged from v2.

The first four phases contain 12 parsed, correlated MCP calls. The forget
stream contains six further tool results, which the original gate did not
finish parsing. Its preview response was 50,711 bytes, containing 19 candidate
facts and 22 additional source events. Claude replaced that inline JSON with
a persisted-output notice and a partial preview. The strict parser rejected
the notice with `Claude tool result is not JSON`.

Reading the saved full output afterward explains additional product failures;
it does not repair the model's original view or make this run pass. Claude
selected one additional fact and source. The first apply request omitted
characters from the source ID and was rejected without committing. A second
request corrected that ID and committed. This was not the v2 mistake of adding
the target's automatically covered source IDs.

The committed receipt explicitly retained six exact copies, twelve advisory
facts and twenty-one eligible source events. A read-only database scan still
found forgotten labels in six fact rows, eight source events and six evidence
rows. Therefore accepting the persisted-output wrapper in the parser would
not satisfy the forgetting gate. The complete requested deletion was not
established, even though the selected subset committed successfully.

The [diagnosis](native-claude-v3-diagnosis.json) binds the raw trace and saved
preview, request/result lines and final database scan. The [receipt](native-claude-v3-receipt.json)
binds 138 verified private archive members, SHA-256
`8b3ead28e90c49fbbe3db3d751b27d60793bb224df737e73bb4358d5d84f3c23`.
Owned processes are absent, SQLite integrity and foreign-key checks pass,
the service descriptor and copied credentials were removed, and both source
credential files are unchanged. No live configuration changed.

The next repair needs a bounded, complete review path for large previews and
long source identifiers. Preserve the preview digest, explicit selection,
scope, concurrency and deletion-coverage guards. Never hide omitted candidates
or treat an incomplete review as complete. Prototype and test that contract
before freezing another native attempt. Do not weaken the parser or increase
the phase budget to recast this failure.

This is one synthetic attempt. It establishes neither general reliability nor
pilot readiness. The private two-project pilot has not started.
