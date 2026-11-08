# Category coverage and seed adjudication

This increment adds 14 cases and 22 target decisions. The development additions cover conflicts that arise only where schedule validity intervals overlap, inclusive date boundaries, dual signatures combined with expiry, and a hypothetical visit date different from the reference date. Tuning additions qualify competing quota values by ratification before determining whether they conflict. All scenarios are synthetic and already inspected; none can become fresh acceptance data.

Independent review inspected all 14 additions. It required changing the schedule facts and queries to say scheduled departure, preserving the distinction between a schedule and an observed event. Those corrections are applied. It found no mandatory inherited-template link at the complete-construction level: date-dependent conflict, expiry short-circuiting an unknown signature, and authority qualification add structure beyond the earlier seed. This does not prove independent semantic populations.

A separate end-to-end review inspected all 51 original seed requests and 255 target decisions, including every owned source. It found three remaining rule-text ambiguities: Larch stated only necessity while its source also supplied sufficiency; Ibis and Kestrel stated sufficiency while their sources also supplied necessity. All three target rules now explicitly say if and only if. The reviewer verified those amendments and found no remaining binary, reason, temporal or attribution correction. The amended seed is preserved as `.venv/joint-support-seed-v5`; v4 remains unchanged.

Per-case review status is now recorded for the seed, the previously reviewed 12-case expansion, the previously reviewed 16-case audit partition draft, and these 14 additions. This closes pending label adjudication for the current cases; it does not certify corpus sufficiency. The corrected combined artifact is `.venv/joint-support-coverage-v2`:

| Partition | Cases | Declared connected groups | Supported target occurrences |
| --- | ---: | ---: | ---: |
| Training | 63 | 3 | 37 |
| Tuning | 12 | 3 | 7 |
| Development | 18 | 4 | 15 |

There are 334 total target labels. All 93 cases are marked reviewed. The mechanical minimum partition check now passes because each partition has positive and negative labels and none are pending or disputed. The audit still reports semantic_labels_verified=false and training_ready=false: the program cannot certify the human-readable semantic adjudication or dataset sufficiency.

Known remaining data gates include source-ownership traps and source-embedded instructions in independent development constructions, explicit critical-fact criteria and coverage, and a justified account of the training distribution's heavy wrong-subject/relation imbalance. The model's capacity and bounded training recipe remain separate gates. More rows alone cannot close any of these requirements.

The corrected inputs, labels and audit reproduce exactly from the builders and match their saved hashes. Retained-tokenizer serialization preserves every payload and decision marker; current lengths range from 105 to 552 tokens. These short examples do not define the production token budget or prove feasibility on longer retained sources. No retained model weights, training or live configuration were used.

| Corrected combined artifact | SHA256 |
| --- | --- |
| inputs.json | 2e3804bfc6f6933a3a123fc295bc13eee74558a83f21c8a5060a76bc9d41665d |
| labels.json | 7f6ed1a5446f66ad79f22afb75b63387384e31290485a958c9fe3e334cb02524 |
| structural-audit.json | eb38d0b0f66e7cf6f46b4da57c4106cc77943fc85b0427f8eda069376393749c |

V1 remains the uncorrected, pending-review artifact with 93 cases and 334 labels. Its input SHA256 is `103cf52445713bec269483573e66c206f0aa92109a45b11d482321b7008c7761`; its label SHA256 is `e089373ab4d95794ccb4dc220453b18f42e9f01897d82b8c7451283b99d0756f`. Neither version changes any previous model score, quality gate, timing target, original pilot requirement or publication state.
