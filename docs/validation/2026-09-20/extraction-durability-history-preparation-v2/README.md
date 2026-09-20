# Durability and history development comparison: v2 preparation

Frozen before execution: 46 reused cases, two arms, one repetition, 92 calls.
The candidate adds qualification preservation and response-boundary guidance
on top of the first durability clause. Production remains unchanged. Source
cc940ac is package-verified; sleep.py bytes match both prior experiment sources.

The 26 durability cases retain their original rubric and ambiguity review.
The 20 historical/preference cases retain their source events, unrelated
existing facts and original rubric/ambiguity review. Unused stale criterion
fields from the old runner inputs are omitted; no semantic label was changed.
Alternating arm order, model digest, source hashes and provider settings are
frozen in the preparation. This is reused development data, not unseen proof.

The candidate addresses observed loss of revocation conditions and retention
of a one-time prohibition as a project policy. Risks include suppressing real
negative requirements, deliverable-specific requirements, and useful historical
reports. All 46 cases must be reviewed, including failures and ambiguities.
The candidate is not adopted. No facts are committed by this experiment.

Offline preflight passed all 92 case/arm paths using a stub response, checking
schemas, scopes, prompt differences and equal structured options. It made zero
model generation calls and provides no semantic performance evidence.

Launch only after the first durability run is terminal, archived and reviewed.
This preparation does not record a started or completed model run.
