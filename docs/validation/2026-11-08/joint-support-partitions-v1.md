# Draft tuning and development constructions

This increment authors 16 cases across four scenario families. Proposed tuning families combine reused asset aliases with serial-number specifications, and a general rule with a scoped exception. Proposed development families distinguish exclusive choices from conjunctions and preferences, and a nested report from its subsequent retraction and an actual decision. They are authored audit data, not fresh acceptance cases.

The first draft has 79 examples and 311 target labels: 63 training, eight tuning and eight development examples. The structural audit finds three training, two tuning and two development connected groups under the declared ancestry. All labels remain pending. These counts are not an independent review of template separation or label correctness.

Coverage remains inadequate for training readiness. Development has no unresolved-condition, temporal-mismatch or unresolved-conflict labels. Tuning has no attribution or unresolved-conflict labels. Neither partition establishes critical-fact coverage. The mechanical minimum partition check remains false because final review is incomplete. Passing that minimum later would still not establish sufficient diversity.

The retained tokenizer serialized all 79 draft requests without changing their payloads or redirecting marker positions. Their lengths range from 105 to 549 tokens. The counting helper used a 100,000-token ceiling only to inventory lengths; this is not the prospective deployment budget or a capacity result. No model weights were loaded. The corpus's short inputs cannot justify truncating longer runtime evidence.

Draft v1 artifacts are retained under `.venv/joint-support-partitions-v1`:

| Artifact | SHA256 |
| --- | --- |
| inputs.json | fde5c80e328c333b4c40fd110e4d3f66700197144d25c01a515acebc81d152f2 |
| labels.json | defb92340d7f329a9b3c5a7918434071296baeb1bd53e1699f4f02645f22adba |
| structural-audit.json | 6fbfe2816b147658b89c667ab0244c9bf02d4e81b88a18971e249c83cd6948ed |

The tokenizer binding remains `620982bd901e79a30f6025cddf6b67a65560b4e83c8732431530231252cc728e`. The inventory records input, serializer and helper hashes alongside every frame hash. Label and ancestry review is pending; corrections must create a separate artifact version.

The first review found three concrete issues. The date-boundary question relied on a current fact and an undated serial specification; the corrected fact and source explicitly cover October 1 through November 8. The older serial's specification now also states its September validity. A historical speech label relied only on the speaker's later claim that she had said something earlier; a separate earlier transcript now establishes that speech event. Finally, the purported reordered cohort dropped one candidate; the corrected variant retains all four candidates and only changes their order. V1 is preserved as the rejected draft.

Final bounded review found no further concrete correction in the 16 cases. It found no mandatory inherited-template link at the complete-construction level: alias joins, scoped exceptions, exclusive alternatives and retraction chronology add structure beyond the earlier seed. The groups remain provisional, not proven independent populations. Labels remain pending for final corpus adjudication and coverage readiness remains false.

The corrected v2 artifact has 79 examples and 312 target labels, with the same 63/8/8 split. The additional target restores the complete permutation. Retokenization of v2 again spans 105 to 549 tokens and preserves all payloads and marker bindings.

| V2 artifact | SHA256 |
| --- | --- |
| inputs.json | 465813316c4372d3c4a221706f16e83d2cd7cda0910b64549e44ddcf0f800e88 |
| labels.json | 97f09ec3205fd58700e654483299dc917335143342f9641a7f3e91ed58984004 |
| structural-audit.json | c392beb2b1ab96395ce0451209c13331352d83ac42f713f1553e6cc1e45a94d0 |

All v2 objects were reproduced from the builder and matched saved hashes. Earlier seed/expansion inputs and annotations remain unchanged. No model inference, training, capacity preflight, product integration or live pilot change occurred.
