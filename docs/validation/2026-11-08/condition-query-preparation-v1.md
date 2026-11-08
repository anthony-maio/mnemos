# Condition-query probe preparation

The 16-case condition probe is prepared, not executed. Its 30 predicate judgments test whether the question establishes a requested circumstance, with source conditions supplied separately. The provider inputs contain the question and predicate catalog; expected states and applicability remain in a separate grading file. This is authored development data following earlier failures, not independent acceptance.

The serial runner records full owner requests, prompts, schemas, provider replies and decoded circumstances. It continues after completed malformed replies without retrying, but stops on uncertain or unconfirmed provider completion. Journal failures and revision fences prevent advancing to another request. Cancellation is recorded and propagated. The runtime owner must still verify identity, token limits, provider termination and cleanup; the runner alone does not establish those properties.

All 68 selected condition tests passed in 1.46 seconds, including ten serial-runner cases. JUnit is `.venv/condition-query-runner-v1.xml`. Black and diff checks passed. Runner source is committed at `6eedc15a03923bea70b37b3dfc35db44c5d752ff`.

Frozen files are in `.venv/condition-query-probe-development-v1`. Their SHA256 values are:

| File | SHA256 |
| --- | --- |
| execution.json | 2f20a1568ccb1f311296ac902c4b9774fbfc14c84ad323f4abec854e97c6eb8c |
| grading.json | e71fa2b4b9a1abdee601c12075fb649b54cb3f0bb6779b1b1a5fa7d8854352f4 |
| policy.json | 2a619605c26844e92227553c5bb2341bd27bff73f7cee12c9a614da9076dfd17 |

The policy binds source hashes and the retained Qwen3:14b model/runtime identities. Planned execution uses context 8192, output limit 192 tokens, temperature 0, seed 2026110809, raw prompts without thinking, one call per case and a 15-second cooperative timeout. There are no semantic warmup calls or retries. These settings bound the experiment; they do not assert that the output limit will suffice or the model will meet the original timing requirement.

Before inference, the runtime wrapper must verify these files and the exact model, executable and DLL hashes, reconstruct the saved owner requests without regenerating identifiers, and confirm rendered prompts match saved bytes. Grade protocol validity, predicate states, polarity, exact witness binding, semantic witness adequacy and final applicability separately. Alternate valid quotations require review; an exact reference match is not the only correct witness, and an exact substring is not sufficient proof of entailment.

The separate token-cost result remains a timing warning. This probe cannot satisfy source extraction, complete retrieval latency, fresh acceptance or the original two-week pilot. No production integration, publication or live configuration change occurred.
