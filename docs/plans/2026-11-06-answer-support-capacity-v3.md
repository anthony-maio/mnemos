# Answer-support capacity comparison v3

Status: prospective development protocol. Freeze the source and this protocol
before inference. The earlier v1/v2 experiments and thresholds remain unchanged.

## Question

The 4B model's complete, valid probability distributions still selected answers
that lacked the requested attribute, used the wrong subject or environment, or
lost temporal and uncertainty qualifications. Several wrong choices had
probability 1.0. Test whether a larger model improves these decisions using the
same support questions and the same frozen retrieval inputs. Do not change
prompts, thresholds or labels during this comparison.

## Models and support treatments

Use Ollama 0.40.9 and these existing Qwen3 Q4_K_M artifacts:

- `qwen3:4b`, digest
  `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`.
- `qwen3:14b`, digest
  `bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`.

The [official 14B model listing](https://ollama.com/library/qwen3:14b) describes
the 14.8B Q4_K_M artifact. The actual downloaded digest and model metadata must
match the local preflight. Model size alone is not evidence of better support.

Each model receives both existing candidate-at-a-time treatments:
`joint_v1` and `separate_v2`. Require every choice probability, unit sum within
0.01 and selection of a highest-probability option. Keep the support threshold
at 0.8, the shared operation deadline at 15 seconds and the serialized request
limit at 64,000 bytes. Failures and timeouts receive no abstention credit.
Do not interpret self-reported probabilities as calibrated confidence.

## Identical accepted facts and retrieval inputs

Use the unchanged development dataset
`benchmarks/local_memory/fixtures/answer-support-development-v1.json`, SHA-256
`2c388cb8a603d4a1463496e0a8243a0a22496026c2a0e1d42dca640ad05d9aef`.
It contains ten synthetic scenarios and twenty reused questions: ten answerable
and ten without a supported answer. This is not fresh or held-out evaluation.

Seed each scenario once with distinct local and foreign Git projects. Disable
extraction and automatic enrichment. Prepare vector, lexical, hybrid and full
retrieval inputs once per query, using isolated copies of the same accepted
SQLite snapshot. Keep BGE small English v1.5 embeddings, five-candidate maximum,
scope visibility and the 1,200-unit conservative byte context budget fixed.
The full comparison arm explicitly enables spreading; affective routing stays
off. These settings do not change application defaults.

Freeze the returned IDs, facts, grades, context accounting and snapshot hash
before either model runs. Every model/treatment reads those same IDs against
the original unchanged accepted store. Gold IDs stay in grading records and
never enter judgment requests. Verify original SQLite snapshot hashes before
and after model blocks, and require identical input hashes within every paired
query/retrieval group. A mismatch invalidates the study.

This gives 80 frozen retrieval inputs and 320 observations: two models times
two support treatments times four retrieval strategies times twenty questions.
There are still only twenty distinct questions. Do not count treatment or
retrieval combinations as independent samples.

## Runtime, order and evidence

Use the private loopback endpoint on port 11436 and the private evaluation model
cache. The runtime executable SHA-256 is
`0957bdaa7b52e163cdc7feb1275653f27c6b32e8ba309e957bdbb83493e97a4b`.
Set context to 8,192, parallel requests to one, maximum loaded models to one,
temperature zero, thinking off and maximum output to 2,048 tokens. Cap each
rendered prompt at 4,096 UTF-8 bytes, reserving 2,048 output and 2,048 template
positions under the conservative byte bound. Verify actual token counts and
absence of truncation after execution. These context settings differ from v2;
use the new paired 4B observations as the comparison baseline.

Randomize model order once with seed 20261106 and run one model block at a time.
Warm-load before measurement, verify the active model digest and context, and
unload between blocks. Randomize treatment order per frozen input with the
recorded deterministic seed. Record GPU occupancy and actual model/VRAM sizes;
CPU offload or concurrent device activity limits latency interpretation. Do not
stop another application's model to improve these measurements.

Use a new output directory. Bind source commit, all source hashes, protocol,
dataset, model digests and runtime settings before observations. Preserve each
raw request and response with its model/treatment/request ID, including invalid
responses and missing replies. Record every attempted observation. Preserve
failed attempts and do not tune or restart them in place. Unload the model and
stop only the private server owned by this comparison after execution.

## Decision

Report all four treatments across all four retrieval arms: expected-answer
hits, clean hits, correct abstentions, unsupported returned facts, failures,
timeouts, request counts, provider token counts and support latency. Separate
retrieval candidate coverage from answer support. Audit each remaining semantic
failure against its retained source, including explicit negative answers,
subject/environment mismatches, old versus current facts and missing attributes.

The 90% answerable-recall and 90% abstention targets remain. Passing this small
development comparison does not promote a default. Fresh evaluation with at
least 200 scenarios and three repetitions, separate extraction evaluation,
supported context efficiency, actual daily-use pilot and consistent delivery
remain required. The independently completed native explicit-memory lifecycle
does not replace those gates.
