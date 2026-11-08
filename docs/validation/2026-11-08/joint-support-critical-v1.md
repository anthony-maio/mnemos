# Critical action constraints and adversarial grouping

This increment defines critical action constraints before any model outcomes. A fact qualifies when its own reviewed evidence establishes an explicit prohibition, necessary prerequisite, authorization boundary or operational conformance boundary, and losing that constraint could make a prohibited or nonconforming action appear permitted. This is a declared subset of important memory, not a replacement definition for the original goal's critical facts.

The frozen registry contains ten facts: seven training, one tuning and two development. Each entry binds the target identity, version, exact content, assertion status and complete owned evidence through a full-candidate hash, with a criterion and justification. Changes to any of those fields invalidate the binding. Ordinary colors, equipment IDs, capacities and schedules are outside this criterion, as are unsupported approver or permission claims. Exclusion does not prove that such facts could never be critical under another explicit criterion.

Criticality is independent of query relevance. All occurrences of a registered fact carry the flag, but only adjudicated supported query-target pairs enter the critical selection-retention denominator. There are 18 such training opportunities, two tuning opportunities and five development opportunities. These 25 occurrences are not 25 independent critical facts. An unavailable supported opportunity must count as a miss; selecting an irrelevant critical fact still harms precision. Eventual scoring must report each fact's applicable outcomes, not count a fact as successful because it was selected once. No outcomes or retention score are reported here.

Six new tests cover source/content/version/status invalidation, missing registry entries, unique-fact versus opportunity counts, exclusion of unresolved queries from the supported denominator and preservation of original annotations. Review confirmed the source bindings and denominator counts, but caught an unconditional reviewed status in the corpus manifest writer. The writer now derives that status from the audit; two additional tests prevent pending or disputed cases from receiving a reviewed manifest. Together with the data-audit suite, 25 tests passed. The current v1 artifact was accurately labeled because all 101 cases were reviewed. The code emits measured=false and does not claim extraction retention, durable storage retention or completion of the original critical-fact gate.

The adversarial draft adds eight reviewed cases. Independent review found their labels defensible but rejected two coverage claims:

- Forged assistant/tool output in source text extends the earlier instruction-in-source template. All four related cases are training-only and explicitly inherit `expansion-source-instruction`; role labels and JSON syntax do not justify a separate development group.
- The four source-ID collision cases exercise namespace collisions and subject/claim checking. No foreign source establishes the exact unsupported Unit Y claim, so the negatives do not isolate ownership enforcement. They remain development namespace fixtures and do not close the source-ownership gap.

The assembled draft contains 101 cases and 351 target labels, split 67 training, 12 tuning and 22 development. Declared connected groups are 3/3/5. All cases have per-case review recorded, and mechanical partition minimums pass. Independent development ownership and embedded-instruction coverage remain open, as do training-distribution justification, retained-model capacity and the frozen training recipe. The original requirement for at least 100 critical facts in fresh acceptance remains unchanged.

Artifacts are retained under `.venv/joint-support-corpus-v1`:

| Artifact | SHA256 |
| --- | --- |
| inputs.json | 4d986f648a34251f6bc795b0050dbe47f42753263d1364aeeab9e1e0cd07f5d2 |
| labels.json | aedcfeb5314dc3efa8d57c924f7e790774aad880d4474d2ca6501295e7ce9773 |
| structural-audit.json | 7cd23945d9713b0708680a0e6bae5f4ef89b896cff2203fc33094a8010fe1da7 |
| critical-registry.json | ba5f238c973c469b29eae8945c61830929bc7e1b10a52120e07b6486aadb59d3 |
| critical-inventory.json | eea0bc50ff90d559f4ce64c113446e3933fff47b538b706aeca44dc6ed5f3289 |

Earlier artifacts remain separate. No training, retained-model inference, live memory/configuration change or pilot update occurred.
