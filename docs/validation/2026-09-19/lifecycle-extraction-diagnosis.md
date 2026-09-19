# Why the local lifecycle matrix has missing correction targets

This read-only diagnosis concerns the preserved `fc72cc2` run in the
[terminal lifecycle archive](lifecycle-incomplete-local-v2/README.md).
No source event, gold label, threshold, response or checkpoint was changed.

All 50 partial observations contain exactly one correction error:
`missing_accepted_target`, with no accepted fact evidence matching the target.
The preceding captures committed, maintenance completed, and the surprisal
gate kept the events. The extractor explicitly returned valid JSON with
`decision="no_durable_facts"` and an empty fact list. The subsequent assistant
events also produced no-fact decisions. Across those 100 structured responses,
there were no parse, evidence-ID, proposal-validation or firewall errors.
Generation ended with `done_reason="stop"`; extraction output used at most
132 tokens against a 512-token cap. Initial extraction prompts were
1,453-1,482 characters and 367-383 evaluated prompt tokens. The traces provide
no truncation signal.

The pattern is concentrated in the 20 correction scenarios. Their initial
automatic captures use the form "The earlier project note said ...". Across
the five sleep-enabled profiles and three repetitions, 250 of 300 such
captures produced one fact proposal; 50 produced none. The no-fact reasons
describe the source as historical information that does not establish a
current preference or continuing requirement. Baseline and the profile
without sleep directly create proposals and do not exercise that extraction
decision, explaining their lack of this particular failure.

The prompt in [sleep.py](../../../mnemos/modules/sleep.py) requires direct
source support, preserves assertion provenance, and warns that quotations
do not prove a preference or completed action. Its reference to "current
source events" governs evidence for support/correction; it is not an explicit
ban on storing historical information. The model's temporal interpretation
is observed in its reasons. Whether discarding the historical assertion is
the desired product behavior is not established by this experiment.

This points to an ambiguity between the fixture's expected acquisition of a
superseded fact and the model's interpretation of a report about an earlier
note. It does not prove that all 50 extraction decisions were correct, or
that the product should treat historical claims as current facts. Historical
information may require an attributed, temporally qualified representation.
Raw event retention and accepting a present-tense fact are different actions.

The development correction scenario starts with a direct assertion. The
[v3 generator](../../../benchmarks/local_memory/generate_lifecycle_v3.py)
preserves the older held-out source wording while adding explicit target
metadata. In [the adapter](../../../benchmarks/local_memory/ollama_adapter.py),
target resolution requires exactly one accepted fact supported by the declared
source event; it correctly refuses to guess an ID when no such fact exists.
The original reports remain incomplete under their original contract.

There is also a checkpoint-accounting limitation: partial observation IDs
are retained in `completed_observation_ids`, and resume skips them. Retrying
only the no-fact outcomes would select for successful extraction and bias the
comparison. It would not repair the original experiment.

Next work should first establish a development contract for direct assertions,
attributed historical statements, and unavailable correction targets. A
deterministic no-fact response must leave the fact store unchanged and make
correction fail without a fabricated target; a directly asserted accepted
fact must remain correctable. Keep execution coverage, product rejection,
failed quality outcomes, and missing measurements distinct. Any subsequent
experiment needs a prospective protocol and fresh output binding. Preserve
the original held-out labels and failures; do not tune the extractor against
these observed case IDs or reclassify missing targets as successful correction.
