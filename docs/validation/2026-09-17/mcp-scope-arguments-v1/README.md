# MCP scope argument regression

This archive records the five stdio MCP cases for the `allowed_scopes`
transport contract. The string form and list form containing `project` must
retrieve the project fact while excluding a foreign-project fact. Empty lists,
unknown scope names, and non-string list members must fail validation.

The first red run is retained as an invalid fixture run: all five cases failed
because the fixture omitted `session_id`. The v2 red run corrected that fixture
and exposed the actual server schema mismatch: one case failed while four
passed. The first green run included this five-case parametrization plus five
compatibility cases; nine passed and the legacy string case timed out before
MCP startup. The unchanged rerun records all ten green cases passing. The
copied logs and XML are the executed outputs, and the test source is the exact
final formatted source used for the green runs. Red v1 omitted session_id;
red v2 corrected it before the Black formatting pass.

The archive binds the regression to source before the product change
`0b5de1a422290abf298070c1132c5ab318e06aec` and after the product change
`d58e082f9a2cec1c88d50504603cc0db43ca8a96`. This is local stdio MCP and
SQLite fixture evidence. It does not establish native-host, model, provider,
or production-network behavior. The archive does not include credentials,
live configuration, or database files.

The archive-local `.gitattributes` preserves the copied log and XML bytes.
`evidence-index.json` records exact byte counts and SHA256 hashes for every
other archived file.
