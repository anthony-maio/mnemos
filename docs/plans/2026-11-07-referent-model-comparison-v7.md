# Referent model comparison v7

Status: prospective. Commit source, fixtures and this protocol before inference.

Compare the unchanged referent comparator on Qwen3:14b Q4_K_M and
Qwen3.5:9b-q4_K_M. The earlier 19/20 operation score hid relation errors, and
the 72-view diagnostic exposed sensitivity to target IDs, order and distractors.
This development screen asks whether changing the model improves those results.
It cannot establish fresh quality, production readiness or pilot completion.

Run each model once on each of these frozen datasets, reported separately:

- `reconciliation-development-v1.json`, 20 cases, SHA-256
  `366ae35e01f517b88a0826dc2f109d51775065d552fcd0eb992033196776fef0`.
  Use the single `referent_only_v7` arm, with the existing operation labels.
- `referent-stability-development-v1.json`, 72 views of twelve seed scenarios
  in five clusters, 198 labeled relations, SHA-256
  `d69661dafbe6c8099dd5a110fcecdce5f2848d1a868f4fe964d42aaf831b21c5`.
  Use `relations_v6`, with operation, relation and stability scoring unchanged.

The second dataset extends the development material. Do not pool its views
with the first dataset as independent or fresh observations. Preserve all
earlier scores; explicit sampling makes this a new run of both models.

Model identities:

| Model | Manifest digest | Model blob SHA-256 |
| --- | --- | --- |
| qwen3:14b | bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8 | a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e |
| qwen3.5:9b-q4_K_M | 56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90 | 02d45dc1cf451ba2475ac33b301c2dd8f985abe4c182ce04a1f2f5bf0260278d |

Use Ollama 0.40.9, the generate route, thinking false, temperature zero,
top_k 20, top_p 0.95, min_p 0, repeat_penalty 1, presence_penalty 0,
frequency_penalty 0 and seed 2026110701. These controls are explicit on every
request. Retain each model's own template, tokenizer and stop handling; this
compares model packages and does not isolate weights alone. Zero temperature
and a seed do not establish reproducibility across hardware or runner versions.
The options contract follows the [Ollama API](https://docs.ollama.com/api/generate)
and [versioned option definitions](https://github.com/ollama/ollama/blob/v0.40.9/api/types.go).

Use a private loopback runtime with one loaded model and one request slot.
Load Qwen3 first, run the 20-case screen then the 72-view diagnostic, unload it,
then repeat that sequence with Qwen3.5. Use the same full schedule shuffle,
seed 2026110701, within each dataset. Sequential loading and cache order limit
the latency comparison; report measured latency without a speedup claim.

Keep context 8192, output allowance 4096, counted input cap 3584, template
reserve 512, provider deadline 90 seconds and observation deadline 120 seconds.
Native prompt-plus-schema counting must use the loaded model's tokenizer.
Reject model overrides that would invalidate that binding. Verify manifest,
model blobs, executable hashes, loaded model identity, native process ownership,
native model path, context and slot count before measurement and after each run.

Before quality inference, require one synthetic structured-response smoke check
per model with a fixed literal answer, no emitted thinking and normal stop.
Require repeated token IDs and exact detokenization for five synthetic strings
including whitespace, escaped Unicode and literal special-token notation.
Count all captured development prompts against the cap before starting each
model's quality run. Smoke checks are setup evidence and receive no quality
credit. Preserve a failed smoke check and stop that model's run without trying
a different route or prompt inside this study.

Before running, test sampling forwarding, unchanged default payloads, invalid
controls, model/digest selection and tokenizer mismatch rejection. Compare
preflight prompts and schemas with the frozen v5/v6 artifacts. Keep expected
labels and logical target aliases out of requests. Commit all bound code and
data. Record original responses, settings, tokens, snapshots, timings, errors
and stop reasons. Do not overwrite or silently retry a completed observation.
Automatic transient HTTP retries remain visible in the response records.

Report exact operations, correct review outcomes, wrong accepted targets,
wrong accepted additions and operational failures. For the 72-view diagnostic,
also report relation accuracy, combined exactness, all-view correctness by seed
and stability after mapping IDs to logical targets. Consistently wrong outputs
remain wrong. Independently replay recorded final responses through validation
and rebuild scores, source hashes and input hashes. Compare matched cases and
views between models, retaining every failure in its original denominator.

A candidate can proceed to a separately frozen confirmation only if it makes
all 20 original decisions correctly and all 72 diagnostic decisions with all
198 relations correct, with no operational failures or unsafe acceptance.
Passing this development gate would still not count as fresh acceptance.

Archive private evidence, unload both models and verify owned processes and
ports are closed. No live configuration, database writes, retrieval, extraction
quality measurement, native-host session, pilot day or publication is involved.
