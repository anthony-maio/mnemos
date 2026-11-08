# Claude batch-inspection lifecycle v7

The full attempt failed. The forget phase reached its 180-second timeout;
batch inspection did not establish a complete lifecycle within the existing
bounds. No retry, corrective prompt or budget increase was used.

The runner was frozen at `25ce0945aebdbcda3f5ef59475bc5d7ba7c36891`.
Its full regression passed 4,537 tests with 89 skips and no failures or errors
in 801.64 seconds. This is the runner worktree's suite, not a full-suite result
for the distinct delivery checkout. Delivery source was
`de45f13b92b3bf7a43bd65c9220ec9fe71b154d4`; its installed wheel SHA-256 was
`db1fff9cd7a74f174477a06e35928d884c199c327b8f9b56649487bca55680e5`.
The wrapper verified 99 installed package files and 57 dependency versions,
the installed smoke receipt, source bindings and passing regression before
launching the host.

| Phase | Result |
| --- | --- |
| Teach | Passed, 3 correlated calls, reported cost $0.037392 |
| Restart recall | Passed, 2 correlated calls, reported cost $0.0473025 |
| Correct | Passed, 5 correlated calls, reported cost $0.0717498 |
| Corrected recall | Passed, 2 correlated calls, reported cost $0.0575304 |
| Reviewed forgetting | Failed: 180-second phase timeout |
| Final restart, source replay and recall | Not attempted |

The forget process receipt records `timed_out=true`, 181.313 seconds including
termination, return code 1, removed authentication copy and no cleanup errors.
The runner's generic error says "Native process or authentication cleanup
failed"; this does not establish an authentication cleanup failure. The process
receipt identifies the timeout, and both cleanup layers report success.

The raw partial stream contains one retrieve call, one target inspection, five
preview calls and three batch-inspection calls. Those batch calls request the
same seven IDs, starting without a cursor and then requesting offsets 30 and
62. There is no `mnemos_forget_related` invocation or terminal result in the
stream. No final forget-phase cost is available.

An independent offline audit reconstructed all five preview pages: 92 fragments
with a matching snapshot digest. The seven-fact batch also reconstructed
completely across three pages and 83 fragments. Its ordered IDs, versions and
contents match preview candidates f1-f7, classified as `exact_assertion`.
Candidates f8-f19 are `similarity_candidate`; they were neither inspected nor
selected by an apply call. All 22 source events were eligible. There is no
evidence here of a paging defect or of an unsafe deletion selection.

Read-only SQLite inspection found all 23 pre-forget fact contents and versions
unchanged, retained controls intact, zero suppressions and a passing integrity
check. Deletion, source suppression and final restart recall remain unperformed.

The CLI MCP log reports 653 milliseconds across the ten tool calls: 149 for
retrieve, 19 for inspect, 176 across five preview pages, and 309 across three
batch pages. The interval from the last preview completion to the first batch
invocation was 99.827 seconds, with no tool call. That interval includes host,
model and network time; it is not a measurement of pure model reasoning.
These observations argue against treating service execution as the main cause
of this timeout. The next investigation should measure the cost of presenting
and interpreting review material before proposing another native attempt.

The outer job ended after 338.199 seconds and reports zero remaining processes.
Credential copies were removed, watched profiles were unchanged, all bound
inputs were unchanged, and runner HEAD and tracked source remained unchanged.
No live configuration or optional pilot installation was upgraded.

Private evidence is under
`C:/Users/Ant/AppData/Local/Temp/mnemos-improvement-checks-20261106/native-claude-v7-de45f13b`.
The wrapper control receipt is `.venv/native-claude-batch-v7-control.json`;
the outer process receipt is `.venv/native-claude-batch-v7-job/process.json`.
The bound source and selected raw receipts were preserved in
`.venv/native-claude-batch-v7-evidence.zip`: 146 files plus a manifest,
SHA-256 `4b75550cdc32b15b7aa43499ccd6556bea192b7b31c9edfa38b589fe07145a93`.
Executable binaries are excluded and retain their hashes in the manifest.

Keep this attempt failed. Audit the completed and missing review material
before proposing a different interface. This run establishes neither a cost
nor a latency benefit from batch inspection. Semantic quality, latency gates,
actual pilot days and publication remain open.
