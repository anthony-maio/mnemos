Extraction contract development v1 archive

This directory preserves the original eight-call development diagnostic before
the v2 schema changes. It is not heldout evidence and it does not establish
semantic extraction quality. The run used four fixed cases once against the
a38 frozen source and once against the candidate source, with gemma3:4b at
temperature 0 and a 512 token limit.

Model and source provenance:

- Ollama endpoint: http://192.168.86.35:11434
- Model: gemma3:4b
- Model digest: a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a
- Frozen source hash: 60970620b87eaed2cf035b515d53188bc8855feaac77c03f529389d4765842c5
- Candidate source hash: 8e388f9fbfe7a526806d2f338e070af112d110eb0e656fb7e9f01bb265c78fa3
- Runner SHA256: 9CB7E2A9A5BD9B411DBC39F48F4686FDC55FE366226C2515BCD861E576838DCD
- Report SHA256: B354A5A5F4C8BBA465FF6F08C27BF182A9949A582A8B49D79105A9021B436A63

The predefined exact case proxy compared the first parsed proposal with the
fixed expected operation, content, evidence, and target. Version rejection was a runtime check. It is a
contract proxy, not an independent semantic judgment. All proposals in every
raw response were manually reviewed for this archive:

- Frozen supported_existing_fact: one exact support proposal, accepted.
- Candidate supported_existing_fact: one exact-content add proposal. It did
  not satisfy the expected support operation.
- Frozen explicit_correction: one add proposal correctly citing the source,
  plus one support proposal citing the existing fact ID as evidence. Runtime
  rejected the response for unavailable evidence.
- Candidate explicit_correction: one correct proposal with source evidence and
  the existing target, but expected_version was 2 while the target version was
  1. Runtime rejected the stale version.
- Both profiles returned no_durable_facts without the required reason for the
  unrelated_new_fact and question_no_fact cases. Runtime preserved these as
  invalid responses.

The source-bound candidate schema produced request-specific evidence ID enums.
The v1 schema did not bind target IDs or target/version pairs. The raw Ollama
responses and provider metadata remain in report.json and raw/*.json. No facts
were persisted; each child used an in-memory store. The run made no heldout or
production claim and must not be rerun as a tuned result.
