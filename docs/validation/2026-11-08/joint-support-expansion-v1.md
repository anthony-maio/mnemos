# Support corpus expansion

The new builder adds 12 training examples and 20 target labels to the preserved seed. The combined draft has 63 examples and 275 target labels. New constructions cover a conjunction supported across two owned sources, evidence attached to the wrong target, multiple selected targets, a reordered cohort, attributed reports versus sensor measurements, negative prerequisites, explicit hypotheticals, and source-embedded instructions.

Each new label includes a case-specific rationale. All labels remain pending. All examples are training-only; this increment supplies no tuning, development or fresh acceptance data. The audit's six connected groups reflect declared ancestry and exact-content overlap, not proof of six independent semantic templates. The corpus still has no designated critical facts, and it lacks sufficient independent development coverage. No training or model inference ran.

Validation: the serializer and data-audit suite passed 35 tests. The original seed builder still reproduces the v4 inputs and labels exactly. The combined inputs, labels and structural report were written to `.venv/joint-support-expansion-v1`, and all manifest hashes were verified from disk. The structural report correctly leaves minimum partition checks, semantic verification and training readiness false.

Draft artifact bindings:

| Artifact | SHA256 |
| --- | --- |
| inputs.json | 3033017e943c568ec50e85c56fcf8cd0f57995de2ebc6b61ede5cc0b014b3cdb |
| labels.json | 7c1a9108904cbe00a8284792809d1c4b4584aeafdd7c93450bb734ac7196582e |
| structural-audit.json | 4d28d9a269dc03f66800d71241eb74873bbbf47324c670c4f6a074a10188a75f |

The independent review identified two ambiguities: the relay target stated only a necessary condition while its source supplied sufficiency, and the sensor record did not explicitly bind the sensor to Unit K. The builder now states the relay's complete if-and-only-if rule and names the measured unit. A supported report that cannot answer an independent measurement question now carries attribution_mismatch instead of incomplete_source_support.

Review also rejected treating the new condition, attribution and partial-conjunction constructions as independent of the seed. Their template lineages now connect to the existing groups. V2 has three connected training groups, not six. The third group contains source-instruction cases; this metadata count still does not certify semantic independence or adequate diversity. All review statuses remain pending for final corpus adjudication.

The corrected artifacts are preserved separately in `.venv/joint-support-expansion-v2`:

| Artifact | SHA256 |
| --- | --- |
| inputs.json | 03ce1a6de33e217e511a4788e6673d42c18de8e2fdd59a4777fb54fd2031d624 |
| labels.json | 5bd776df2bcaa7bebfcd9578c0684945e07921a08384c7e6201663b887f111df |
| structural-audit.json | 74f2f647377ff156c938e125b8878eff4a1f871f2cd4a82a76ee21ac0fb492c6 |

Neither version is approved training data. Independent tuning/development families, temporal boundary cases, implicit conflicts, capacity preflight and a frozen training recipe remain outstanding. The original quality, timing, fresh acceptance and pilot requirements remain open.

Final bounded review found no remaining concrete label error in the 12 corrected expansion cases and verified the v2 corrections. This supports those cases under the written policy; it does not certify the full corpus, group independence or training readiness. V2 inputs, labels and audit were reproduced exactly from the final builder, and all saved hashes matched.
