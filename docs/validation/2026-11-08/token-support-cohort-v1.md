# Separate token-support development cohort

The authored cohort contains 12 synthetic scenarios, 25 candidate facts, 26 retained source excerpts, and 24 questions. Twelve questions are answerable and twelve require empty abstention. All twelve expected answer facts are marked critical for the development retention check. Gold labels and rationales are stored separately from model inputs.

The cases cover direct support, missing facts, pending proposals, conditional requirements, future effective dates, explicit replacement, unresolved peer disagreement, fictional quotations, unsupported peer claims, source instructions, partial-source overclaims, numeric state, and disagreement among a fact's own sources. These are authored development cases, not independent annotation, extracted facts, live project data, or fresh acceptance evidence.

Structural preflight generated all 52 source-witness requests. Every request retains its complete comparison cohort. The largest final prompt is 2,321 UTF-8 bytes, and each complete question fits the selector's request-count and total-byte limits. Label IDs reference existing candidates and critical IDs are subsets of expected IDs. No model was called.

The first preparation used descriptive scenario/fact identifiers in source metadata. Before any inference, a second preparation replaced those model-visible identifiers with opaque scenario, fact, and evidence IDs, and remapped the separate grading file. This avoids exposing category hints such as unsupported-peer. The first preparation is retained; no measured result was replaced. Question keys remain orchestration metadata and are not inserted into prompts.

The prepared snapshots explicitly represent synthetic accepted project facts. They are not a replay of a live store or an extraction-quality test. The runtime must preserve their exact bytes and fingerprint bindings; subsequent retrieval evaluation will require actual equal accepted-store snapshots and separate extraction evidence.

Actual tokenizer preflight and the full inference-runtime freeze are still pending. The prepared artifacts are not yet authorized by the protocol to run as a semantic screen. Next, bind the exact runner and dependencies, verify every complete prompt's token count without truncation, and retain the lexical question order with each candidate's source order. Run each question once, with no retries or threshold changes. An unrelated warmup, if used, must be declared before execution and kept separate. Any operational failure remains a failure, not an abstention.

The fixed selector rule and all original external quality, complete latency, capture, and pilot requirements remain unchanged. The model-visible input file contains no expected-ID or critical-ID fields; the grading file must not be parsed by the inference runner.
