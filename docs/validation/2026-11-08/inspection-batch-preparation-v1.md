# Batch inspection preparation

V6 selected ten additional facts without the required inspection and exhausted
its phase budget. A new batch-inspection draft in canonical delivery accepts
one to twenty explicitly ordered fact IDs. Each uses the existing scoped
inspection request. Any unavailable or mismatched result rejects the batch;
no partial successful output is returned. The tool never selects facts for
deletion.

Complete results are presented as lossless JSON fragments under the existing
12,000-byte wrapped-output bound. The snapshot digest covers every result,
including versions and evidence, and changes or order changes invalidate a
continuation cursor. Focused batch, existing forget-paging and MCP wiring tests
pass: 48 tests. The tool is not yet part of native acceptance or an installed
candidate. Independent review and contract support requiring complete page
coverage for every selected fact remain outstanding.

A separate experimental string-sharing codec was evaluated but is not the
production format. All 13 codec tests pass, including reserved-key injection,
malformed references and round trips. Applied to the nine actual V6 inspection
responses, it reconstructs the original results exactly. Their original text
was 30,397 bytes; a compact JSON list is 26,792 bytes; string sharing produces
23,943 bytes, a 10.63% reduction versus the compact list. Forty distinct strings
were shared. These byte savings do not establish model comprehension, token
cost, complete review, or native success. The ten missing inspections were not
invented or counted as evaluated results. The production draft uses ordinary
lossless fragments instead of this additional reference format.

Private measurement: `.venv/inspection-batch-measurement-v1/result.json`.
Input trace SHA-256:
`9ddcee101cbc49032d65c4bcf8dadf8f5662d8947bc19ceffdf73f16c50cab6d`.
Codec SHA-256:
`951a60d2ad5cce2ec78536635c19208d2d719016f3e63d4d897f1ef829075c2d`.
V6 remains failed. No additional model run or budget increase occurred.
