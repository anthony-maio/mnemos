# Qwen structured extraction control diagnostic

This archive contains two four request, candidate only development diagnostics
using the same four prompts and request schemas. Neither run is heldout data,
production evidence, or an integration claim.

## Provenance

- Model: `qwen3.5:9b`
- Model digest: `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`
- Endpoint: `http://192.168.86.35:11434`
- Candidate revision: `a27ad6e5b7fb4227bb66eeb9737f5ca62eb45975`
- Candidate source SHA256: `ebd2e00e0508d12df9b60562d2f9adc841dd0bea1699e5bd187842a3dfc45b5a`
- Fixed temperature: `0.0`
- Fixed `num_predict`: `512`

## Default qwen run

The default provider call made four Ollama requests with the stored structured
format. All four calls ended with `done_reason=stop`, but all had an empty
`response` and a separate nonempty `thinking` field. The parser therefore
recorded 0/4 parseable responses and 0/4 schema or runtime validations. The
reasoning field was retained as raw provider output and was never treated as an
answer.

## Direct think false control

The direct `/api/generate` control replayed each stored prompt and format
exactly, adding only top-level `think:false` and the required non-streaming
transport/options fields. It produced 4/4 parseable responses, 4/4 request
schema validations, 4/4 response schema validations, and 4/4 runtime model
validations.

The exact contract proxy passed 2/4 cases:

- Supported existing fact: passed with one support proposal, source evidence,
  and the existing target ID.
- Explicit correction: failed because the model emitted `add` rather than
  `correct`, even though the PostgreSQL content and source evidence were
  present.
- Unrelated new fact: failed because the model emitted no proposals instead of
  the expected add proposal.
- Question: passed with no durable facts and a reason.

The exact proxy checks all returned proposals and fixed fields. It is not a
semantic precision measurement. Semantic entailment remains unjudged. The
correction case also has a design limitation: the source says the user now
requires PostgreSQL while the old fact says the project uses SQLite. That is a
requirement versus current state distinction, so an add result does not by
itself prove that correction was wrong. A future correction diagnostic needs a
disjoint explicit supersession case.

Raw responses, prompts, schemas, request hashes, response byte hashes, and
run manifests are preserved in the two subdirectories. No facts were written.
