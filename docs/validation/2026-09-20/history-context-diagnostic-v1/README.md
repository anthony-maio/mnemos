# Existing-fact extraction context diagnostic

Forty-two real Qwen2.5:14b proposal observations compare the same seven
source events with and without the unrelated existing control fact. Each
condition runs three repetitions, alternating order. Model digest, source
revision, schema-generation code, history clarification, source identities,
and request settings remain fixed. The existing-fact condition verifies its
prompt and structured schema exactly match the prior service request.

| Case | Nonempty proposals with existing fact | Without existing fact |
| --- | --- | --- |
| Historical approval | 3/3 | 3/3 |
| Unverified relay | 3/3 | 3/3 |
| Obsolete archive | 3/3 | 3/3 |
| Fiction | 3/3 | 3/3 |
| Question | 0/3 | 0/3 |
| Hypothetical | 3/3 | 3/3 |
| Assistant suggestion | 3/3 | 0/3 |

There are no request, schema, or provenance errors. Nonempty counts describe
model behavior; they are not semantic accuracy scores. The fiction and
hypothetical failures persist without prior facts. The assistant suggestion
changes behavior when the existing-fact contract is present. That contract
changes both the prompt and target branches in the response schema, so this
does not isolate a prompt-only or schema-only cause.

The instructions prohibit using questions or suggestions as proof of a user
preference or completed action, but allow qualified reports. These outputs
motivate a development clarification separating an actual reported event
from merely recording that fictional, hypothetical, or unadopted content
was mentioned. This is a proposed explanation, not a proven model mechanism.

This diagnostic writes no facts and does not alter the preceding service
failures. It is not held-out evaluation, production quality, or a prompt
deployment. Raw texts and proposals remain available for qualitative review.
The evidence index binds all copied artifacts; this README is outside it.
